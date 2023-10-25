--
-- Medium-level bindings to llama.cpp
--

{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}

module Pomnetic.Medium
  ( Model()
  , Context()
  , PomneticError(..)
  , Token()
  , Batch()
  , SeqID
  , Logits
  , loadModel
  , tokenize
  , bosToken
  , eosToken
  , tokenToText
  , createContext
  , createBatch
  , batchLength
  , setBatchItem
  , setBatchLength
  , getLogits
  , sampleMirostat
  , decode
  , removeTokens
  , BatchItem(..)
  , Logits )
  where

import Control.Exception
import Control.Concurrent
import Control.Monad
import Control.Monad.IO.Class
import Control.Monad.Primitive ( touch )
import Data.Data
import Data.Int
import Foreign.C.String
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.ForeignPtr
import Foreign.Ptr
import Foreign.Storable
import Foreign.C.Types
import Data.Text ( Text )
import qualified Data.Text as T
import Data.Vector ( Vector )
import qualified Data.Vector as V
import System.IO
import System.IO.Unsafe

foreign import ccall "hs_llama_load_model" c_llama_load_model :: CString -> IO (Ptr CModel)
foreign import ccall "&hs_llama_free_model" fptr_c_llama_free_model :: FunPtr (Ptr CModel -> IO ())

foreign import ccall "hs_llama_create_context" c_llama_create_context :: Ptr CModel -> IO (Ptr CContext)
foreign import ccall "&hs_llama_free_context" fptr_c_llama_free_context :: FunPtr (Ptr CContext -> IO ())

foreign import ccall "hs_llama_tokenize" c_llama_tokenize :: Ptr CModel -> CString -> Ptr (Ptr Int32) -> Ptr CSize -> IO CInt
foreign import ccall "hs_free_tokens" c_llama_free_tokens :: Ptr Int32 -> IO ()

foreign import ccall "hs_bos_token" c_llama_bos_token :: Ptr CContext -> IO Int32
foreign import ccall "hs_eos_token" c_llama_eos_token :: Ptr CContext -> IO Int32

foreign import ccall "hs_token_to_text" c_llama_token_to_text :: Ptr CModel -> Int32 -> Ptr CString -> Ptr CSize -> IO CInt
foreign import ccall "hs_free_text" c_llama_free_text :: CString -> IO ()

foreign import ccall "hs_create_batch" c_llama_create_batch :: CInt -> IO (Ptr CBatch)
foreign import ccall "&hs_free_batch" fptr_c_llama_free_batch :: FunPtr (Ptr CBatch -> IO ())

foreign import ccall unsafe "hs_batch_length" c_llama_batch_length :: Ptr CBatch -> IO CInt
foreign import ccall unsafe "hs_batch_capacity" c_llama_batch_capacity :: Ptr CBatch -> IO CInt
foreign import ccall unsafe "hs_set_batch_item" c_llama_set_batch_item
  :: Ptr CBatch
  -> CInt
  -> CInt
  -> CInt
  -> CInt
  -> CInt
  -> IO ()
foreign import ccall unsafe "hs_set_batch_length" c_llama_set_batch_length :: Ptr CBatch -> CInt -> IO ()

foreign import ccall "hs_decode" c_llama_decode :: Ptr CContext -> Ptr CBatch -> IO CInt
foreign import ccall unsafe "hs_get_logits" c_llama_get_logits :: Ptr CContext -> CInt -> Ptr CFloat -> IO ()
foreign import ccall unsafe "hs_get_vocab_size" c_llama_vocab_size :: Ptr CContext -> IO CInt
foreign import ccall "hs_sample_mirostat" c_llama_sample_mirostat :: Ptr CContext -> Ptr CFloat -> Ptr CFloat -> IO Int32

foreign import ccall "hs_remove_tokens" c_llama_remove_tokens :: Ptr CContext -> CInt -> CInt -> CInt -> IO ()

-- Match with llama.cpp
newtype Token = Token Int32
  deriving ( Eq, Ord, Show, Typeable, Storable )

newtype Batch = Batch (ForeignPtr CBatch)
  deriving ( Eq, Ord, Show )

data CContext
data CModel
data CBatch

data Context = Context (MVar (ForeignPtr CContext)) !Model !Int -- vocab size
  deriving ( Eq )

newtype Model = Model (ForeignPtr CModel)
  deriving ( Eq, Ord, Show )

data PomneticError
  = PomneticError String
  | TooLongText
  deriving ( Eq, Ord, Show, Typeable )

instance Exception PomneticError

pomneticError :: MonadIO m => String -> m a
pomneticError msg = liftIO $ throwIO $ PomneticError msg

touchModel :: Model -> IO ()
touchModel (Model model_fptr) = withForeignPtr model_fptr $ \model_ptr ->
  touch model_ptr

withContext :: Context -> (Ptr CContext -> IO a) -> IO a
withContext (Context ctx_mvar model _) action = withMVar ctx_mvar $ \ctx_fptr ->
  withForeignPtr ctx_fptr $ \ptr -> do
    result <- action ptr
    touchModel model
    return result

withModel :: Model -> (Ptr CModel -> IO a) -> IO a
withModel (Model model_fptr) action = withForeignPtr model_fptr action

loadModel :: FilePath -> IO Model
loadModel fpath = mask_ $ do
  raw_model <- withCString fpath $ \fpath_str ->
    c_llama_load_model fpath_str

  when (raw_model == nullPtr) $
    pomneticError $ "Failed to load model: " <> fpath

  fptr <- newForeignPtr fptr_c_llama_free_model raw_model
  return $ Model fptr

createContext :: Model -> IO Context
createContext model = mask_ $ withModel model $ \model_ptr -> do
  raw_context <- c_llama_create_context model_ptr

  when (raw_context == nullPtr) $
    pomneticError "Failed to create context"

  vocab_size <- fromIntegral <$> c_llama_vocab_size raw_context
  fptr <- newForeignPtr fptr_c_llama_free_context raw_context
  mvar <- newMVar fptr

  return $ Context mvar model vocab_size

bosToken :: Context -> Token
bosToken ctx = unsafePerformIO $ withContext ctx $ \ctx_ptr ->
  Token <$> c_llama_bos_token ctx_ptr

eosToken :: Context -> Token
eosToken ctx = unsafePerformIO $ withContext ctx $ \ctx_ptr ->
  Token <$> c_llama_eos_token ctx_ptr

createBatch :: MonadIO m => Int -> m Batch
createBatch sz | sz <= 0 = pomneticError "Batch size must be positive"
createBatch sz = liftIO $ mask_ $ do
  batch <- c_llama_create_batch (fromIntegral sz)
  when (batch == nullPtr) $
    pomneticError "Failed to create batch"
  fptr_batch <- newForeignPtr fptr_c_llama_free_batch batch
  return $ Batch fptr_batch

type SeqID = Int

data BatchItem = BatchItem
  { token :: !Token
  , position :: !Int
  , sequenceId :: !SeqID
  , logits :: !Bool }
  deriving ( Eq, Ord, Show )

decode :: MonadIO m => Context -> Batch -> m ()
decode ctx (Batch fptr_batch) = liftIO $ withForeignPtr fptr_batch $ \ptr -> do
  withContext ctx $ \ctx_ptr -> do
    result <- c_llama_decode ctx_ptr ptr
    when (result > 0) $
      throwIO TooLongText

    when (result < 0) $
      pomneticError "Failed to decode"

removeTokens :: MonadIO m => Context -> SeqID -> Int -> Int -> m ()
removeTokens ctx seq_id start end = liftIO $ withContext ctx $ \ctx_ptr ->
  c_llama_remove_tokens ctx_ptr (fromIntegral seq_id) (fromIntegral start) (fromIntegral end)

setBatchItem :: Batch -> BatchItem -> Int -> IO ()
setBatchItem batch@(Batch fptr_batch) item sz =
  if sz >= len
    then pomneticError $ "Batch size must be less than " <> show len
    else withForeignPtr fptr_batch $ \batch_ptr ->
      let Token tk = token item
       in c_llama_set_batch_item batch_ptr
                                 (fromIntegral sz)
                                 (fromIntegral tk)
                                 (fromIntegral $ position item)
                                 (fromIntegral $ sequenceId item)
                                 (if logits item
                                   then 1
                                   else 0)
 where
  len = batchCapacity batch

batchCapacity :: Batch -> Int
batchCapacity (Batch fptr) = unsafePerformIO $ withForeignPtr fptr $ \ptr ->
  fromIntegral <$> c_llama_batch_capacity ptr

batchLength :: MonadIO m => Batch -> m Int
batchLength (Batch fptr) = liftIO $ withForeignPtr fptr $ \ptr ->
  fromIntegral <$> c_llama_batch_length ptr

setBatchLength :: MonadIO m => Batch -> Int -> m ()
setBatchLength batch@(Batch fptr) len =
  if len > cap || len < 0
    then pomneticError $ "Batch length must be less than " <> show cap
    else liftIO $ withForeignPtr fptr $ \ptr ->
           c_llama_set_batch_length ptr (fromIntegral len)
 where
  cap = batchCapacity batch

vocabSize :: Context -> Int
vocabSize (Context _ _ vocab_size) = vocab_size

type Logits = Vector Float

getLogits :: MonadIO m => Context -> Int -> m Logits
getLogits ctx idx =
  liftIO $ withContext ctx $ \ctx_ptr ->
    allocaArray (vocabSize ctx) $ \logits_ptr -> do
      c_llama_get_logits ctx_ptr (fromIntegral idx) logits_ptr
      V.fromList . fmap cfloatToFloat <$> peekArray (vocabSize ctx) logits_ptr

type Mu = Float

sampleMirostat :: MonadIO m => Context -> Logits -> Mu -> m (Token, Mu)
sampleMirostat ctx logits _mu | V.length logits /= vocabSize ctx =
  pomneticError "Logits size must be equal to vocab size"
sampleMirostat ctx logits mu = liftIO $ withContext ctx $ \ctx_ptr ->
  allocaArray (V.length logits) $ \logits_ptr -> do
    pokeArray logits_ptr (fmap realToFrac $ V.toList logits)
    alloca $ \mu_ptr -> do
      poke mu_ptr (CFloat mu)
      token_idx <- c_llama_sample_mirostat ctx_ptr logits_ptr mu_ptr
      new_mu <- cfloatToFloat <$> peek mu_ptr
      return (Token $ fromIntegral token_idx, new_mu)

cfloatToFloat :: CFloat -> Float
cfloatToFloat = realToFrac

tokenize :: Model -> Text -> Vector Token
tokenize model txt = unsafePerformIO $ withModel model $ \model_ptr -> 
  withCString (T.unpack txt) $ \str -> mask_ $
    alloca $ \tokens_ptr_ptr ->
    alloca $ \tokens_sz_ptr -> do
      result <- c_llama_tokenize model_ptr str tokens_ptr_ptr tokens_sz_ptr
      when (result /= 0) $
        pomneticError "Failed to tokenize"

      ntokens <- peek tokens_sz_ptr
      tokens_ptr <- peek tokens_ptr_ptr

      tokens <- V.generateM (fromIntegral ntokens) $ \idx -> do
        val <- peekElemOff tokens_ptr idx
        return $ Token $ fromIntegral val

      c_llama_free_tokens tokens_ptr

      return tokens

tokenToText :: Model -> Token -> Text
tokenToText model (Token token) = unsafePerformIO $ withModel model $ \model_ptr ->
  alloca $ \str_ptr ->
  alloca $ \str_len_ptr -> mask_ $ do
    result <- c_llama_token_to_text model_ptr token str_ptr str_len_ptr
    when (result /= 0) $
      pomneticError "Failed to convert token to text"

    str <- peek str_ptr
    str_len <- peek str_len_ptr

    result <- peekCStringLen (str, fromIntegral str_len)
    c_llama_free_text str

    return $ T.pack result
