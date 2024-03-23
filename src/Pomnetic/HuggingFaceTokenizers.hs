{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE CPP #-}

module Pomnetic.HuggingFaceTokenizers
  ( HFTokenize(..)
  , hfTokenize
  , hfTokenizeEmpty
  , tokenizeByHF
  , tokenToTextByHF
  , bosTokenByHF
  , eosTokenByHF
  , sepTokenByHF
  )
  where

import Control.Monad
import Data.Aeson
import qualified Data.ByteString.Lazy as BL
import Data.Text ( Text )
import qualified Data.Text as T
import qualified Data.Text.Encoding as T
import qualified Data.Text.IO.Utf8 as T
import Data.Vector ( Vector )
import qualified Data.Vector as V
import Control.Concurrent
import Control.Exception
import Data.Maybe
import Foreign.C.Error
import Foreign.C.Types
import GHC.Generics
import Pomnetic.Safe
import Pomnetic.Types
import Pomnetic.PyHuggingFaceTokenizers ( pycode )
import System.IO
import System.IO.Temp
import System.IO.Unsafe
import System.Process
import System.Process.Internals
import System.Timeout

foreign import ccall "hs_llama_kill_pid" c_llama_kill_pid :: CLong -> IO CInt

-- Used with Python to recognize lines that are part of the protocol.
magic :: Text
magic = "dQ25CNJGb94QejK0"

-- -- Behind the scenes, we manage one interpreter where we run Python. This
-- interpreter is locked by a mutex and then all calls with tokenizeByHF are
-- run in that interpreter.
pyInterpreterProcess :: MVar (Maybe (ProcessHandle, Handle, Handle))
pyInterpreterProcess = unsafePerformIO $ newMVar Nothing
{-# NOINLINE pyInterpreterProcess #-}

withPyInterpreter :: (Handle -> Handle -> IO a) -> IO a
withPyInterpreter f = do
  result <- mask $ \restore -> modifyMVar pyInterpreterProcess $ \old -> do
      (interpreter, istdin, istdout) <- case old of
        Nothing -> launchInterpreter
        Just x -> return x

      result <- try $ restore $ f istdin istdout

      return (Just (interpreter, istdin, istdout), result)
  case result of
    Left (exc :: SomeException) -> throwIO exc
    Right x -> return x
 where
  launchInterpreter :: IO (ProcessHandle, Handle, Handle)
  launchInterpreter = mask_ $ withSystemTempFile "pomnetic_pyglue.py" $ \temp_file_path temp_file_handle -> do

    T.hPutStr temp_file_handle pycode
    hClose temp_file_handle

    (Just istdin, Just istdout, _, phandle) <- createProcess
      (proc "python" [temp_file_path])
        { std_in = CreatePipe
        , std_out = CreatePipe
        , std_err = Inherit
        }

    flip onException (killProcess phandle >> waitForProcess phandle) $ do
      result <- timeout 60000000 $ waitForReady istdout
      when (isNothing result) $ do
        killProcess phandle
        exit_code <- waitForProcess phandle
        throwIO $ userError $ "Python interpreter did not start in time and report readiness. Exit code: " <> show exit_code

      return (phandle, istdin, istdout)

  waitForReady :: Handle -> IO ()
  waitForReady h = do
    line <- T.strip <$> T.hGetLine h
    if line == (magic <> " READY")
      then return ()
      else waitForReady h

-- Kill process.
--
-- The process library only has terminateProcess, which sends SIGTERM. What a
-- wussy. We want to send SIGKILL.
killProcess :: ProcessHandle -> IO ()
killProcess phandle =
#if defined(WINDOWS)
  -- does windows have SIGKILL equivalent?
  terminateProcess phandle
#else
  withProcessHandle phandle $ \p_ -> case p_ of
    ClosedHandle{} -> return ()
    OpenExtHandle{} -> return ()
    OpenHandle pid -> do
      throwErrnoIfMinus1Retry_ "killProcess" $ c_llama_kill_pid (safeFromIntegral pid)
#endif

data PyTokenizerAnswer = PyTokenizerAnswer
  { tokens :: [Int]
  }
  deriving (Eq, Show, Generic, FromJSON, ToJSON)

-- | Invoke a HuggingFace tokenizer.
--
-- To use HF tokenizers, you must have a Python interpreter and 'transformers'
-- installed usable by that interpreter.
--
-- The tokenizer will be kept in memory so that further calls with the same
-- model are fast.
tokenizeByHF :: HFTokenize -> IO (Vector Token)
tokenizeByHF tokenize = withPyInterpreter $ \stdin stdout -> do
  let req = object [ "type" .= ("tokenize" :: Text)
                   , "model" .= hfModelID tokenize
                   , "add_special_tokens" .= addSpecialTokens tokenize
                   , "local_files_only" .= localFilesOnly tokenize
                   , "trust_remote_code" .= trustRemoteCode tokenize
                   , "text" .= textToTokenize tokenize
                   ]

  let req_encoded = encode req
  BL.hPut stdin req_encoded
  hFlush stdin
  BL.hPut stdin "\n"
  hFlush stdin

  waitForAnswer stdout
 where
  waitForAnswer :: Handle -> IO (Vector Token)
  waitForAnswer h = do
    line <- T.strip <$> T.hGetLine h
    if T.isPrefixOf magic line
      then do
        let answer = T.drop (T.length magic+1) line
        case decodeStrict' $ T.encodeUtf8 answer of
          Just (PyTokenizerAnswer tokens) -> do
            return $ V.fromList $ fmap intToToken tokens
          _ -> throwIO $ userError $ "Invalid answer from Python: " <> show answer
      else waitForAnswer h

newtype PyTokenToTextAnswer = PyTokenToTextAnswer
  { py_token_to_text :: Text
  }
  deriving (Eq, Show, Generic, FromJSON, ToJSON)

-- | Convert a token into text.
tokenToTextByHF :: FilePath -> Token -> IO Text
tokenToTextByHF model_id token = withPyInterpreter $ \stdin stdout -> do
  let req = object [ "type" .= ("token_to_text" :: Text)
                   , "model" .= model_id
                   , "token" .= tokenToInt token
                   ]

  let req_encoded = encode req
  BL.hPut stdin req_encoded
  hFlush stdin
  BL.hPut stdin "\n"
  hFlush stdin

  waitForAnswer stdout
 where
  waitForAnswer :: Handle -> IO Text
  waitForAnswer h = do
    line <- T.strip <$> T.hGetLine h
    if T.isPrefixOf magic line
      then do
        let answer = T.drop (T.length magic+1) line
        case decodeStrict' $ T.encodeUtf8 answer of
          Just (PyTokenToTextAnswer text) -> do
            return text
          _ -> throwIO $ userError $ "Invalid answer from Python: " <> show answer
      else waitForAnswer h

-- | Default HF tokenizer. This has local_files_only=False and trust_remote_code=False.
--
-- Also, no special tokens are added automatically.
hfTokenize :: FilePath -> Text -> HFTokenize
hfTokenize path text_to_tokenize = HFTokenize {
    hfModelID = path
  , localFilesOnly = False
  , trustRemoteCode = False
  , textToTokenize = text_to_tokenize
  , addSpecialTokens = False
  }

-- | Tokenize an empty string. Usable in `setModelToHFTokenizer`.
hfTokenizeEmpty :: FilePath -> HFTokenize
hfTokenizeEmpty path = hfTokenize path ""

-- Gets the BOS token used by the HF model.
bosTokenByHF :: FilePath -> IO (Maybe Token)
bosTokenByHF model_id = do
  metadata <- metadataByHF model_id
  return $ intToToken <$> py_bos_token metadata

eosTokenByHF :: FilePath -> IO (Maybe Token)
eosTokenByHF model_id = do
  metadata <- metadataByHF model_id
  return $ intToToken <$> py_eos_token metadata

sepTokenByHF :: FilePath -> IO (Maybe Token)
sepTokenByHF model_id = do
  metadata <- metadataByHF model_id
  return $ intToToken <$> py_sep_token metadata

data PyTokenizerMetadata = PyTokenizerMetadata
  { py_bos_token :: Maybe Int
  , py_eos_token :: Maybe Int
  , py_sep_token :: Maybe Int
  }
  deriving (Eq, Show, Generic, FromJSON, ToJSON)

metadataByHF :: FilePath -> IO PyTokenizerMetadata
metadataByHF model_id = withPyInterpreter $ \stdin stdout -> do
  let req = object [ "type" .= ("metadata" :: Text)
                   , "model" .= model_id
                   ]

  let req_encoded = encode req
  BL.hPut stdin req_encoded
  hFlush stdin
  BL.hPut stdin "\n"
  hFlush stdin

  waitForAnswer stdout
 where
  waitForAnswer :: Handle -> IO PyTokenizerMetadata
  waitForAnswer h = do
    line <- T.strip <$> T.hGetLine h
    if T.isPrefixOf magic line
      then do
        let answer = T.drop (T.length magic+1) line
        case decodeStrict' $ T.encodeUtf8 answer of
          Just metadata@(PyTokenizerMetadata{}) -> do
            return metadata
          _ -> throwIO $ userError $ "Invalid answer from Python: " <> show answer
      else waitForAnswer h
