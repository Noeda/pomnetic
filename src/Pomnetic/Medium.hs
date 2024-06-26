{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE LambdaCase #-}

--
-- Medium-level bindings to llama.cpp
--
-- How to understand the process:
--
-- First, load a model with `loadModel`. llama.cpp takes `.gguf` files these
-- days.
--
-- Then, create a context of that model with `createContext`.
--
-- You use contexts to generate text. `Context` has a lock on it, so using it
-- from multiple threads will block one until one is done.
--
-- To generate text, you have to create batches. Use `createBatch` to make a
-- batch. You give the batch a capacity, which will tell how many tokens it may
-- contain at maximum. Batches start out empty (capacity and length are not the
-- same thing).
--
-- Fill the batch with tokens using `setBatchItem`. These items may be from one
-- or multiple independent text generation tasks. For the items you set their
-- position ID, token ID and sequence ID they are from. You can use `tokenize`
-- and other token functions to figure out which tokens to put in the items.
-- For the last token you want to set `logits = True` for the batch item, which
-- will instruct the system to predict probabilities for the token that would
-- come after that token.
--
-- Use `processBatch` to instruct the system to process the batch. Afterwards,
-- you can use `getLogits` to get probabilities for your tokens. (or
-- `getLogitsMutable`)
--
-- You can use `sampleMirostat` to sample a token from the logits. (or you can
-- also do sampling yourself). For next generation, you could make a batch with
-- just one item, the new token you just generated and set logits = True to
-- predict next tokens. The context internally has a cache that remembers all
-- previous tokens, so they don't need to be included in the batch again.
--
-- Use `forgetTokens` to alter what the context remembers about the tokens. You
-- can use this to regenerate portion of the text, or just have the context
-- forget your text entirely.
--

{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}

module Pomnetic.Medium
  (
  -- * Models
    Model()
  , loadModel
  , loadModelWithHFTokenizer
  -- * Contexts
  , createContext
  , Context()
  , ContextSettings(..)
  , getContextLengthFromMetadata
  , makeSensibleDefaultContextSettings
  , makeSensibleDefaultContextSettingsFromContextLength
  , forgetTokens
  -- * Errors
  , PomneticError(..)
  -- * Tokenization
  , Token()
  , tokenize
  , bosToken
  , eosToken
  , bosTokenModel
  , eosTokenModel
  , tokensToText
  , tokenToInt
  , intToToken
  , vocabularySize
  -- * Filtering
  --
  -- Filtering is used to constraint what output the LLM can generate.
  , Filters()
  , andFilters
  , orFilters
  , regexFilter
  , RegexFilterText
  , attoparsecFilter
  , attoparsecBSFilter
  -- * Batching, and output
  --
  -- The key to efficient LLM text generation is batching.
  , BatchItem(..)
  , Batch()
  , createBatch
  , batchLength
  , setBatchItem
  , BatchItemIdx
  , setBatchLength
  , processBatch
  , SeqID
  -- * Sampling
  , Logits
  , MLogits
  , getLogits
  , getLogitsMutable
  , SoftmaxLogitsable(..)
  , sortLogits
  -- ** Mirostat sampling
  , MirostatConfig(..)
  , mirostatConfig
  , MirostatState()
  , makeMirostatState
  , cloneMirostatState
  , mirostatMu
  , MirostatMu
  , sampleMirostat )
  where

import Control.Exception
import Control.Concurrent
import Control.Monad
import Control.Monad.IO.Class
import Control.Monad.Primitive ( touch )
import Data.Data
import Data.Int
import Data.IORef
import Data.Maybe
import Data.Traversable
import Data.Word
import GHC.Generics
import Foreign.C.String
import Foreign.Marshal.Alloc
import Foreign.Marshal.Utils
import Foreign.ForeignPtr
import Foreign.Ptr
import Foreign.Storable
import Foreign.C.Types
import Data.Attoparsec.Text
import qualified Data.Attoparsec.ByteString as B
import Data.Ord
import Data.Text ( Text )
import qualified Data.Text as T
import qualified Data.Text.Encoding as T
import Data.Unique
import qualified Data.Vector.Unboxed as VU
import qualified Data.Vector.Unboxed.Mutable as VUM
import qualified Data.Vector.Algorithms.Heap as VUA
import Pomnetic.Error
import Pomnetic.HuggingFaceTokenizers
import Pomnetic.Safe ( safeFromIntegral )
import Pomnetic.Types
import System.IO.Unsafe
import Text.Regex.Base.RegexLike
import Text.Regex.TDFA.Text

foreign import ccall "hs_llama_load_model" c_llama_load_model :: CString -> CInt -> IO (Ptr CModel)
foreign import ccall "&hs_llama_free_model" fptr_c_llama_free_model :: FunPtr (Ptr CModel -> IO ())

foreign import ccall "hs_llama_read_context_length_from_metadata" c_llama_read_context_length_from_metadata :: Ptr CModel -> IO CInt

foreign import ccall "hs_llama_create_context" c_llama_create_context
    :: Ptr CModel
    -> CInt
    -> CInt
    -> CInt
    -> CInt
    -> IO (Ptr CContext)
foreign import ccall "&hs_llama_free_context" fptr_c_llama_free_context :: FunPtr (Ptr CContext -> IO ())
foreign import ccall "hs_llama_tokenize" c_llama_tokenize :: Ptr CModel -> CString -> Ptr (Ptr Int32) -> Ptr CSize -> IO CInt
foreign import ccall "hs_free_tokens" c_llama_free_tokens :: Ptr Int32 -> IO ()

foreign import ccall "hs_bos_token_model" c_llama_bos_token_model :: Ptr CModel -> IO Int32
foreign import ccall "hs_eos_token_model" c_llama_eos_token_model :: Ptr CModel -> IO Int32

foreign import ccall "hs_token_to_text" c_llama_token_to_text :: Ptr CModel -> Int32 -> Ptr CString -> Ptr CSize -> IO CInt
foreign import ccall "hs_free_text" c_llama_free_text :: CString -> IO ()

foreign import ccall "hs_create_batch" c_llama_create_batch :: CInt -> IO (Ptr CBatch)
foreign import ccall "&hs_free_batch" fptr_c_llama_free_batch :: FunPtr (Ptr CBatch -> IO ())

foreign import ccall unsafe "hs_batch_length" c_llama_batch_length :: Ptr CBatch -> IO CInt
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
foreign import ccall unsafe "hs_get_logits_from_hs_batch_ptr" c_llama_get_logits_from_hs_batch_ptr :: Ptr CBatch -> CInt -> IO (Ptr CFloat)
foreign import ccall unsafe "hs_batch_has_logits" c_hs_batch_has_logits :: Ptr CBatch -> CInt -> IO CInt
foreign import ccall unsafe "hs_get_logits_len_from_hs_batch" c_hs_get_logits_len_from_hs_batch :: Ptr CBatch -> CInt -> IO CSize
foreign import ccall unsafe "hs_get_vocab_size" c_llama_vocab_size :: Ptr CContext -> IO CInt
foreign import ccall unsafe "hs_get_vocab_size_model" c_llama_vocab_size_model :: Ptr CModel -> IO CInt
foreign import ccall "hs_sample_mirostat" c_llama_sample_mirostat :: Ptr CContext -> Ptr CMirostatState -> CFloat -> CFloat -> IO Int32
foreign import ccall "hs_get_mirostat_logits" c_hs_get_mirostat_logits :: Ptr CMirostatState -> IO (Ptr CFloat)
foreign import ccall "hs_get_mirostat_blacklist" c_hs_get_mirostat_blacklist :: Ptr CMirostatState -> IO (Ptr Word8)
foreign import ccall "hs_make_mirostat_logits_size" c_hs_make_mirostat_logits_size :: Ptr CMirostatState -> CSize -> IO CInt
foreign import ccall "hs_set_mirostat_mu" c_hs_set_mirostat_mu :: Ptr CMirostatState -> CFloat -> IO ()
foreign import ccall "hs_get_mirostat_mu" c_hs_get_mirostat_mu :: Ptr CMirostatState -> IO CFloat

foreign import ccall "hs_remove_tokens" c_llama_remove_tokens :: Ptr CContext -> CInt -> CInt -> CInt -> IO ()

foreign import ccall "hs_create_mirostat_state" c_llama_create_mirostat_state :: IO (Ptr CMirostatState)
foreign import ccall "&hs_free_mirostat_state" fptr_c_llama_free_mirostat_state :: FunPtr (Ptr CMirostatState -> IO ())

vocabularySize :: Model -> Int
vocabularySize model = unsafePerformIO $ withModel model $ \model_ptr -> do
  vocab_size <- c_llama_vocab_size_model model_ptr
  return $ fromIntegral vocab_size

-- Batch uniq capacity mvar
data Batch = Batch !Unique !Int (MVar (ForeignPtr CBatch))

instance Eq Batch where
  Batch uniq1 _ _ == Batch uniq2 _ _ = uniq1 == uniq2

instance Ord Batch where
  Batch uniq1 _ _ `compare` Batch uniq2 _ _ = uniq1 `compare` uniq2

instance Show Batch where
  show (Batch uniq sz _) = "<#Batch " <> show (hashUnique uniq) <> " " <> show sz <> ">"

data CContext
data CMirostatState
data CModel
data CBatch

data Context = Context (MVar (ForeignPtr CContext)) !Model !Int -- vocab size
  deriving ( Eq )

contextModel :: Context -> Model
contextModel (Context _ model _) = model

data Model = Model {
    modelForeignPtr :: !(ForeignPtr CModel)
  , hfTokenizer :: !(IORef (Maybe HFTokenize))
  }

instance Show Model where
  show model = "<#Model " <> show (modelForeignPtr model) <> ">"

instance Eq Model where
  m1 == m2 = modelForeignPtr m1 == modelForeignPtr m2

instance Ord Model where
  m1 `compare` m2 = modelForeignPtr m1 `compare` modelForeignPtr m2

pomneticError :: MonadIO m => String -> m a
pomneticError msg = liftIO $ throwIO $ PomneticError msg

touchModel :: Model -> IO ()
touchModel model = withForeignPtr (modelForeignPtr model) $ \model_ptr ->
  touch model_ptr

withContext :: Context -> (Ptr CContext -> IO a) -> IO a
withContext (Context ctx_mvar model _) action = withMVar ctx_mvar $ \ctx_fptr ->
  withForeignPtr ctx_fptr $ \ptr -> do
    result <- action ptr
    touchModel model
    return result

withModel :: Model -> (Ptr CModel -> IO a) -> IO a
withModel model action = withForeignPtr (modelForeignPtr model) action

-- | Same as `loadModel` but uses a HuggingFace tokenizer.
--
-- Use `hfTokenizeEmpty` from Pomnetic.HuggingFaceTokenizers to make the
-- `HFTokenize` argument. It would take the model_id you would normally use in
-- AutoTokenizer.from_pretrained in HuggingFace Python.
loadModelWithHFTokenizer :: MonadIO m => FilePath -> HFTokenize -> m Model
loadModelWithHFTokenizer fpath hftokenize = do
  model <- loadModel fpath
  setModelToHFTokenizer model hftokenize
  return model

loadModel :: MonadIO m => FilePath -> m Model
loadModel fpath = liftIO $ mask_ $ do
  -- TODO: set 10000 (number of gpu layers to be configurable)
  raw_model <- withCString fpath $ \fpath_str ->
    c_llama_load_model fpath_str 10000

  when (raw_model == nullPtr) $
    pomneticError $ "Failed to load model: " <> fpath

  hf_tokenizer_ref <- newIORef Nothing

  fptr <- newForeignPtr fptr_c_llama_free_model raw_model
  return $ Model {
    modelForeignPtr = fptr
  , hfTokenizer = hf_tokenizer_ref
  }

-- NOT EXPORTED AS PUBLIC API BECAUSE YOU CAN USE IT TO BREAK REFERENTIAL
-- TRANSPARENCY (some functions use unsafePerformIO because they don't expect
-- side effects. This function is called by loadModel instead to prevent
-- setting a HF tokenizer after the fact).
--
-- Makes a model use a HuggingFace tokenizer.
--
-- The user is responsible for making sure the tokenizer is compatible with the
-- model.
--
-- You give it a loaded model, and the a model_id as you would in HuggingFace
-- Python in AutoTokenizer.from_pretrained.
--
-- You can use `hfTokenizeEmpty` from Pomnetic.HuggingFaceTokenizers to make
-- the `HFTokenize` value for this function.
--
-- Once set, all tokenization functions will use the HuggingFace tokenizer.
-- This involves launching a Python process and importing the transformers
-- library, so using a HF tokenizer requires a Python setup.
--
-- This function is lazy and will not launch the Python process until the first
-- token-related function is called.
setModelToHFTokenizer :: MonadIO m => Model -> HFTokenize -> m ()
setModelToHFTokenizer model hftokenize = liftIO $
  writeIORef (hfTokenizer model) (Just hftokenize)

data ContextSettings = ContextSettings
  { contextSettingsMaxTokens :: !ContextLength
  , contextSettingsBatchSize :: !Int
  , contextSettingsNumThreads :: !Int
  , contextSettingsNumBatchThreads :: !Int }
  deriving ( Eq, Ord, Show, Read, Data, Typeable, Generic )

type ContextLength = Int

-- | Given a model, reads out context length out of its metadata. The metadata
-- (usually) tells maximum context length the model can handle coherency;
-- assuming the .gguf file was made to include that properly.
--
-- If the model does not have a context length, or something goes wrong with
-- this, `Nothing` is returned.
getContextLengthFromMetadata :: MonadIO m => Model -> m (Maybe ContextLength)
getContextLengthFromMetadata model = liftIO $ withModel model $ \model_ptr -> do
  context_length <- c_llama_read_context_length_from_metadata model_ptr
  return $ if context_length <= 0
    then Nothing
    else Just $ safeFromIntegral context_length

-- | Given a model, returns sensible default settings.
--
-- This attempts to read context length from the model. Otherwise behaves same
-- as `makeSensibleDefaultContextSettingsFromContextLength`.
makeSensibleDefaultContextSettings :: MonadIO m => Model -> m ContextSettings
makeSensibleDefaultContextSettings model = liftIO $ do
  ctx_length <- getContextLengthFromMetadata model
  makeSensibleDefaultContextSettingsFromContextLength $ fromMaybe 8192 ctx_length

-- | Given a desired context length, returns sensible default settings.
--
-- The default settings use a maximum of 16 CPU cores, otherwise it uses the
-- number of cores available (as reported by `getNumCapabilities`).
makeSensibleDefaultContextSettingsFromContextLength :: MonadIO m => ContextLength -> m ContextSettings
makeSensibleDefaultContextSettingsFromContextLength n_ctx = liftIO $ do
  n_cpus <- max 16 <$> getNumCapabilities
  return $ ContextSettings {
    contextSettingsMaxTokens = n_ctx,
    contextSettingsBatchSize = 512,
    contextSettingsNumThreads = n_cpus,
    contextSettingsNumBatchThreads = n_cpus
  }

createContext :: Model -> ContextSettings -> IO Context
createContext model settings = mask_ $ withModel model $ \model_ptr -> do

  let n_batch = safeFromIntegral (contextSettingsBatchSize settings)
      n_ctx = safeFromIntegral (contextSettingsMaxTokens settings)
      n_threads = safeFromIntegral (contextSettingsNumThreads settings)
      n_batch_threads = safeFromIntegral (contextSettingsNumBatchThreads settings)

  raw_context <- c_llama_create_context model_ptr n_batch n_ctx n_threads n_batch_threads

  when (raw_context == nullPtr) $
    pomneticError "Failed to create context"

  vocab_size <- safeFromIntegral <$> c_llama_vocab_size raw_context
  fptr <- newForeignPtr fptr_c_llama_free_context raw_context
  mvar <- newMVar fptr

  return $ Context mvar model vocab_size

bosTokenModel :: Model -> Token
bosTokenModel model = unsafePerformIO $ do
  ref <- readIORef (hfTokenizer model)
  case ref of
    Nothing -> withModel model $ \model_ptr ->
                 Token <$> c_llama_bos_token_model model_ptr
    Just hf -> bosTokenByHF (hfModelID hf) >>= \case
      Nothing -> throwIO NoRequestedTokenExists
      Just token -> return token

eosTokenModel :: Model -> Token
eosTokenModel model = unsafePerformIO $ do
  ref <- readIORef (hfTokenizer model)
  case ref of
    Nothing -> withModel model $ \model_ptr ->
                 Token <$> c_llama_eos_token_model model_ptr
    Just hf -> eosTokenByHF (hfModelID hf) >>= \case
      Nothing -> throwIO NoRequestedTokenExists
      Just token -> return token

bosToken :: Context -> Token
bosToken ctx = bosTokenModel (contextModel ctx)

eosToken :: Context -> Token
eosToken ctx = eosTokenModel (contextModel ctx)

createBatch :: MonadIO m => Int -> m Batch
createBatch sz | sz <= 0 = pomneticError "Batch size must be positive"
createBatch sz = liftIO $ mask_ $ do
  batch <- c_llama_create_batch (safeFromIntegral sz)
  when (batch == nullPtr) $
    pomneticError "Failed to create batch"
  fptr_batch <- newForeignPtr fptr_c_llama_free_batch batch
  mvar <- newMVar fptr_batch

  uniq <- newUnique
  return $ Batch uniq sz mvar

type SeqID = Int

-- | One item, to be sent to batch.
--
-- `position` tells at which position the item is in the sequence.
--
-- `sequenceId` tells which independent sequence the batch belongs to. The
-- batches can compute multiple independent text generations at once, and the
-- sequence ID is used to distinguish them.
--
-- If `logits` is True, then a prediction is made for the token that should
-- come after this item, and you can use `getLogits` to get the predictions
-- (matching the index at which the BatchItem is in the batch).
data BatchItem = BatchItem
  { token :: !Token
  , position :: !Int
  , sequenceId :: !SeqID
  , logits :: !Bool }
  deriving ( Eq, Ord, Show )

-- | Computes a batch.
--
-- Use `getLogits` to get the logits of a batch, with index matching which
-- index you used in `setBatchItem`.
processBatch :: MonadIO m => Context -> Batch -> m ()
processBatch ctx (Batch _ _ batch_mvar) = liftIO $ do
  withMVar batch_mvar $ \batch_fptr -> withForeignPtr batch_fptr $ \batch_ptr -> do
    withContext ctx $ \ctx_ptr -> do
      result <- c_llama_decode ctx_ptr batch_ptr
      -- FIXME: bad magic number
      when (result == 81273997) $
        pomneticError $ "calloc() failed."
      when (result > 0) $
        throwIO TooLongText

      when (result < 0) $
        pomneticError "Failed to decode"

-- | Makes the context forget tokens in the given range, for the given
-- sequence.
--
-- The range includes the item at given start index, but not the one in end
-- index. I.e. [start, end).
--
-- This understands negative indexes. For example, start=0 and end=-1 will
-- forget all tokens for a sequence.
forgetTokens :: MonadIO m => Context -> SeqID -> Int -> Int -> m ()
forgetTokens ctx seq_id start end = liftIO $ withContext ctx $ \ctx_ptr ->
  c_llama_remove_tokens ctx_ptr (safeFromIntegral seq_id) (safeFromIntegral start) (safeFromIntegral end)

type BatchItemIdx = Int

setBatchItem :: Batch -> BatchItem -> BatchItemIdx -> IO ()
setBatchItem (Batch _ len mvar) item sz =
  if sz >= len
    then pomneticError $ "Batch size must be less than " <> show len
    else withMVar mvar $ \fptr_batch -> withForeignPtr fptr_batch $ \batch_ptr ->
      let Token tk = token item
       in c_llama_set_batch_item batch_ptr
                                 (safeFromIntegral sz)
                                 (safeFromIntegral tk)
                                 (safeFromIntegral $ position item)
                                 (safeFromIntegral $ sequenceId item)
                                 (if logits item
                                   then 1
                                   else 0)

batchLength :: MonadIO m => Batch -> m Int
batchLength (Batch _ _ mvar) = liftIO $ withMVar mvar $ \fptr -> withForeignPtr fptr $ \ptr ->
  safeFromIntegral <$> c_llama_batch_length ptr

setBatchLength :: MonadIO m => Batch -> Int -> m ()
setBatchLength (Batch _ cap mvar) len =
  if len > cap || len < 0
    then pomneticError $ "Batch length must be less than " <> show cap
    else liftIO $ withMVar mvar $ \fptr -> withForeignPtr fptr $ \ptr ->
           c_llama_set_batch_length ptr (safeFromIntegral len)

vocabSize :: Context -> Int
vocabSize (Context _ _ vocab_size) = vocab_size

type Logits = VU.Vector Float
type MLogits s = VUM.MVector s Float

-- | Gets logits for a batch item. The integer refers to the index as used in
-- the last argument of `setBatchItem`.
--
-- Throws `BatchItemIdxHasNoLogits` if the batch item does not have logits.
getLogits :: MonadIO m => Batch -> BatchItemIdx -> m Logits
getLogits (Batch _ _ mvar) idx = liftIO $ withMVar mvar $ \fbatch_ptr -> withForeignPtr fbatch_ptr $ \batch_ptr -> do
  -- c_hs_batch_has_logits checks for range too; returns zero if no logits
  has_logits <- c_hs_batch_has_logits batch_ptr (safeFromIntegral idx)
  when (has_logits == 0) $
    throwIO BatchItemIdxHasNoLogits

  vocab_size <- safeFromIntegral <$> c_hs_get_logits_len_from_hs_batch batch_ptr (safeFromIntegral idx)

  logits_ptr <- c_llama_get_logits_from_hs_batch_ptr batch_ptr (safeFromIntegral idx)

  result <- VUM.new vocab_size
  fill logits_ptr result
  VU.unsafeFreeze result
 where
  fill :: Ptr CFloat -> VUM.IOVector Float -> IO ()
  fill logits_ptr result_vec = go 0 (VUM.length result_vec)
    where
      go !i !len | i >= len = return ()
      go !i !len = do
        !x <- peekElemOff logits_ptr i
        VUM.unsafeWrite result_vec i (cfloatToFloat x)
        go (i + 1) len

-- | Same as `getLogits` but modifies an already allocated mutable vector. The
-- vector must have the same size as the vocabulary.
getLogitsMutable :: MonadIO m => Batch -> BatchItemIdx -> MLogits (VUM.PrimState IO) -> m ()
getLogitsMutable (Batch _ _ mvar) idx mlogits = liftIO $ withMVar mvar $ \fbatch_ptr -> withForeignPtr fbatch_ptr $ \batch_ptr -> do
  -- c_hs_batch_has_logits checks for range too; returns zero if no logits
  has_logits <- c_hs_batch_has_logits batch_ptr (safeFromIntegral idx)
  when (has_logits == 0) $
    throwIO BatchItemIdxHasNoLogits

  vocab_size <- safeFromIntegral <$> c_hs_get_logits_len_from_hs_batch batch_ptr (safeFromIntegral idx)
  when (vocab_size /= VUM.length mlogits) $
    pomneticError "Mutable logits vector size does not match vocabulary size"

  logits_ptr <- c_llama_get_logits_from_hs_batch_ptr batch_ptr (safeFromIntegral idx)

  fill logits_ptr mlogits
 where
  fill :: Ptr CFloat -> VUM.IOVector Float -> IO ()
  fill logits_ptr result_vec = go 0 (VUM.length result_vec)
    where
      go !i !len | i >= len = return ()
      go !i !len = do
        !x <- peekElemOff logits_ptr i
        VUM.unsafeWrite result_vec i (cfloatToFloat x)
        go (i + 1) len


-- | A utility function to take `Logits`, sort them, and return the sorted
-- `Logits`. The sorted list contains an index that determines the token ID.
-- (in `Logits` the token ID is implied by the position of the logit in the
-- vector).
sortLogits :: Logits -> VU.Vector (Token, Float)
sortLogits logits = unsafePerformIO $ do
  thawed <- VU.unsafeThaw $ VU.map (\(x, y) -> (intToToken $ fromIntegral x, y)) $ VU.indexed logits
  -- VUA is currently heap-sort (keeps it stable)
  VUA.sortBy (\(_, a) (_, b) -> compare b a) thawed
  VU.unsafeFreeze thawed

class SoftmaxLogitsable a where
  -- | A utility function to take `Logits` and convert them to probabilities.
  --
  -- Does this operation: `exp(logits) / sum(exp(logits))` (with the exp accuracy
  -- numerical trick of subtracting the maximum value from the logits).
  softmaxLogits :: a -> a

  -- | A utility function to normalize logits to probabilities, so they sum up
  -- to 1. Can be used after e.g. filtering out some logits and maintaining the
  -- relative probabilities and still summing up to 1.
  --
  -- Expects every value to be zero or greater. So don't use on raw logits that
  -- might be negative.
  normalizeLogits :: a -> a

instance SoftmaxLogitsable Logits where
  {-# INLINE softmaxLogits #-}
  softmaxLogits = softmaxLogitsLogits

  {-# INLINE normalizeLogits #-}
  normalizeLogits = normalizeLogitsLogits

instance SoftmaxLogitsable (VU.Vector (Token, Float)) where
  {-# INLINE softmaxLogits #-}
  softmaxLogits = softmaxLogitsTokenFloat

  {-# INLINE normalizeLogits #-}
  normalizeLogits = normalizeLogitsTokenFloat

softmaxLogitsLogits :: Logits -> Logits
softmaxLogitsLogits logits | VU.null logits = VU.empty
softmaxLogitsLogits logits | VU.length logits == 1 = VU.singleton 1.0
softmaxLogitsLogits logits =
  let max_value = VU.maximum logits
      exps = VU.map (\x -> exp (x - max_value)) logits
      -- Normally there is just VU.sum.
      -- sum_exps = VU.sum exps
      sum_exps = dvdSum exps
   in VU.map (/ sum_exps) exps

-- | Same as softmaxLogitsLogits but for (Token, Float) input.
softmaxLogitsTokenFloat :: VU.Vector (Token, Float) -> VU.Vector (Token, Float)
softmaxLogitsTokenFloat logits | VU.null logits = VU.empty
softmaxLogitsTokenFloat logits | VU.length logits == 1 = VU.singleton (fst (VU.head logits), 1.0)
softmaxLogitsTokenFloat logits =
  let max_value = VU.maximumBy (comparing snd) logits
      exps = VU.map (\(tk, x) -> (tk, exp (x - snd max_value))) logits
      sum_exps = dvdSum $ VU.map snd exps
   in VU.map (\(tk, x) -> (tk, x / sum_exps)) exps

normalizeLogitsLogits :: Logits -> Logits
normalizeLogitsLogits logits | VU.null logits = VU.empty
normalizeLogitsLogits logits =
  let sum_logits = dvdSum logits
   in VU.map (/ sum_logits) logits

normalizeLogitsTokenFloat :: VU.Vector (Token, Float) -> VU.Vector (Token, Float)
normalizeLogitsTokenFloat logits | VU.null logits = VU.empty
normalizeLogitsTokenFloat logits =
  let sum_logits = dvdSum $ VU.map snd logits
   in VU.map (\(tk, x) -> (tk, x / sum_logits)) logits

{-# INLINE dvdSum #-}
dvdSum :: (VU.Unbox a, Num a) => VU.Vector a -> a
dvdSum !vec | VU.null vec = 0
dvdSum !vec | VU.length vec == 1 = VU.head vec
dvdSum !vec | VU.length vec == 2 = VU.head vec + VU.last vec
dvdSum !vec | VU.length vec == 3 = (vec VU.! 0) + (vec VU.! 1) + (vec VU.! 2)
dvdSum !vec | VU.length vec == 4 =
  ((vec VU.! 0) + (vec VU.! 1)) +
  ((vec VU.! 2) + (vec VU.! 3))
dvdSum !vec =
  let left_side = VU.take (VU.length vec `div` 2) vec
      right_side = VU.drop (VU.length vec `div` 2) vec
   in dvdSum left_side + dvdSum right_side

-- | Filters constraint what the model can output.
--
-- If the filters block every token, then `AllTokensRejected` will be raised.
data Filters
  = NoFilter
  | AndFilter Filters Filters
  | OrFilter Filters Filters
  | RegexFilter !Text
  | AttoparsecFilter !(Parser ())
  | AttoparsecBSFilter !(B.Parser ())
  deriving ( Typeable, Generic )

andFilters :: [Filters] -> Filters
andFilters [f1] = f1
andFilters [] = NoFilter
andFilters filters = foldl1 AndFilter filters

orFilters :: [Filters] -> Filters
orFilters = foldr OrFilter NoFilter

-- | Adds a filter that demands that the new tokens generated by `generateText`
-- adhere to a PCRE regular expression.
--
-- The regex is evaluated at every token generation, so the string must comply
-- character by character, even when it is not fully generated. This limits the
-- usefulness of the regexes a bit, for example specifying "I want this string
-- to be generated" does not work, because partial matches are not accepted,
-- and the models cannot generate that string in one token.
--
-- You may want to use `attoparsecFilter` instead, that can deal with partially
-- matching input.
--
-- You can however use it to specify character ranges, e.g. "^[^a]*$" will
-- accept text generation that never uses the letter 'a'.
--
-- The regex is only applied to new tokens, the regex will not see past text.
--
-- The regex uses `regex-tdfa` package.
regexFilter :: Text -> Filters
regexFilter regex = RegexFilter regex

-- | Adds a filter that runs an attoparsec. If the parser does not fail, then
-- the token is accepted. Only unambiguous fail is a fail; incomplete input is
-- not considered a fail.
attoparsecFilter :: Parser () -> Filters
attoparsecFilter parser = AttoparsecFilter parser

-- | Same as `attoparsecFilter` but takes bytestrings instead of Text.
--
-- Internally, the output is converted to UTF-8 and then passed as bytestring.
attoparsecBSFilter :: B.Parser () -> Filters
attoparsecBSFilter parser = AttoparsecBSFilter parser

-- | Used in sampling functions to pass in already generated text (to be used
-- with the `regexFilter`).
type RegexFilterText = Text

instance Monoid Filters where
  mempty = NoFilter
  mappend = (<>)

instance Semigroup Filters where
  f1 <> NoFilter = f1
  NoFilter <> f2 = f2
  f1 <> f2 = OrFilter f1 f2

type MirostatMu = Float

data MirostatConfig = MirostatConfig
  { mirostatTau :: !Float
  , mirostatEta :: !Float }
  deriving ( Eq, Ord, Show, Read, Data, Typeable, Generic )

mirostatConfig :: Float -> Float -> MirostatConfig
mirostatConfig tau eta = MirostatConfig tau eta

newtype MirostatState = MirostatState
  { mirostatCStore :: MVar (ForeignPtr CMirostatState) }

makeMirostatState :: MonadIO m => MirostatMu -> m MirostatState
makeMirostatState mu = liftIO $ mask_ $ do
  st <- c_llama_create_mirostat_state
  when (st == nullPtr) $
    pomneticError "Failed to create mirostat state"

  c_hs_set_mirostat_mu st (CFloat mu)

  fptr <- newForeignPtr fptr_c_llama_free_mirostat_state st
  mvar <- newMVar fptr

  return MirostatState { mirostatCStore = mvar }

cloneMirostatState :: MonadIO m => MirostatState -> m MirostatState
cloneMirostatState mstate = liftIO $ withMVar (mirostatCStore mstate) $ \mstate_fptr -> withForeignPtr mstate_fptr $ \mstate_ptr -> do
  st <- c_llama_create_mirostat_state
  when (st == nullPtr) $
    pomneticError "Failed to create mirostat state"

  mu <- c_hs_get_mirostat_mu mstate_ptr
  c_hs_set_mirostat_mu st mu

  fptr <- newForeignPtr fptr_c_llama_free_mirostat_state st
  mvar <- newMVar fptr

  return MirostatState { mirostatCStore = mvar }

mirostatMu :: MonadIO m => MirostatState -> m MirostatMu
mirostatMu mstate = liftIO $ withMVar (mirostatCStore mstate) $ \mstate_fptr -> withForeignPtr mstate_fptr $ \mstate_ptr -> do
  mu <- c_hs_get_mirostat_mu mstate_ptr
  return $ cfloatToFloat mu

sampleMirostat :: MonadIO m => Context -> Logits -> MirostatState -> Filters -> MirostatConfig -> RegexFilterText -> m Token
sampleMirostat ctx logits _mstate _filters _config _regex_filter_text | VU.length logits /= vocabSize ctx =
  pomneticError "Logits size must be equal to vocab size"
sampleMirostat ctx@(Context _ model vocab_size) logits mstate filters config regex_filter_text = liftIO $
  withMVar (mirostatCStore mstate) $ \mstate_fptr -> withForeignPtr mstate_fptr $ \mstate_ptr -> do
    withContext ctx $ \ctx_ptr -> do
      result <- c_hs_make_mirostat_logits_size mstate_ptr (safeFromIntegral $ VU.length logits)
      when (result /= 0) $
        pomneticError "Failed to make mirostat logits"
      mirostat_logits_ptr <- c_hs_get_mirostat_logits mstate_ptr
      blacklist_ptr <- c_hs_get_mirostat_blacklist mstate_ptr

      fillMirostatLogitsPtr mirostat_logits_ptr logits

      test_fun <- fillBlacklist filters blacklist_ptr vocab_size model regex_filter_text

      let sample_token = do token_idx <- c_llama_sample_mirostat
                              ctx_ptr
                              mstate_ptr
                              (CFloat (mirostatTau config))
                              (CFloat (mirostatEta config))

                            when (token_idx == -1) $
                              throwIO AllTokensRejected

                            return token_idx

          try_loop = do token_idx <- sample_token
                        let token = Token $ safeFromIntegral token_idx
                        accepted <- test_fun token
                        if accepted
                          then return token
                          else do pokeElemOff blacklist_ptr (safeFromIntegral token_idx) 1
                                  try_loop

      token <- try_loop
      return token
 where
  fillMirostatLogitsPtr :: Ptr CFloat -> Logits -> IO ()
  fillMirostatLogitsPtr !mirostat_logits_ptr !logits = go 0 (VU.length logits)
   where
    go !idx !len | idx >= len = return ()
    go !idx !len = do
      pokeElemOff mirostat_logits_ptr idx (realToFrac $ VU.unsafeIndex logits idx)
      go (idx + 1) len

{-# NOINLINE fillBlacklist #-}
fillBlacklist :: Filters -> Ptr Word8 -> Int -> Model -> RegexFilterText -> IO (Token -> IO Bool)
fillBlacklist filters ptr vocab_size model regex_filter_text = do
  fillBytes ptr 0 vocab_size
  banlist <- go filters
  return $ \token -> check_filters banlist token
 where
  check_filters :: [Token -> IO Bool] -> Token -> IO Bool
  check_filters [] _token = return True
  check_filters (fun:rest) token = do
    result <- fun token
    if not result
      then return False
      else check_filters rest token

  go :: Filters -> IO [Token -> IO Bool]
  go NoFilter = return []
  go (AttoparsecFilter parser) = do
    return [\token ->
      let token_str = tokensToText model (VU.singleton token)
          whole = regex_filter_text <> token_str
       in do case parse parser whole of
               Fail {} -> return False
               Done {} -> return True
               Partial {} ->
                 -- reject empty tokens for partial matches; prevents the model
                 -- from repeatedly generating an empty token to technically
                 -- adhere to the parser.
                 if T.null token_str
                   then return False
                   else return True]
  go (AttoparsecBSFilter parser) = do
    return [\token ->
      let token_str = tokensToText model (VU.singleton token)
          whole = T.encodeUtf8 $ regex_filter_text <> token_str
       in do case B.parse parser whole of
               Fail {} -> return False
               Done {} -> return True
               Partial {} ->
                 -- reject empty tokens for partial matches; prevents the model
                 -- from repeatedly generating an empty token to technically
                 -- adhere to the parser.
                 if T.null token_str
                   then return False
                   else return True]
  go (RegexFilter regex) = do
    compiled_regex <- case compile defaultCompOpt defaultExecOpt regex of
      Left err -> throwIO $ InvalidRegex (T.pack err)
      Right compiled -> return compiled
    return [\token ->
       let whole = regex_filter_text <> tokensToText model (VU.singleton token)
        in return $ Text.Regex.Base.RegexLike.match compiled_regex whole]

  go (AndFilter f1 f2) = do
    f1s <- go f1
    f2s <- go f2
    return $ f1s <> f2s
  go (OrFilter f1 f2) = do
    f1_funs <- go f1
    f2_funs <- go f2

    let f3_fun token = loopy (f1_funs <> f2_funs) token

        loopy [] _ = return False
        loopy (fun:rest) token = do accepted <- fun token
                                    if accepted
                                      then return True
                                      else loopy rest token

    return [f3_fun]

{-# INLINE cfloatToFloat #-}
cfloatToFloat :: CFloat -> Float
cfloatToFloat !x = realToFrac x

-- | Tokenizes a text. Does not add special tokens (like bos or eos).
tokenize :: Model -> Text -> VU.Vector Token
tokenize model txt = unsafePerformIO $ do
  ref <- readIORef (hfTokenizer model)
  case ref of
    Nothing -> llamaTokenize model txt
    Just hf -> tokenizeByHF hf { addSpecialTokens = False, textToTokenize = txt }
 where
  llamaTokenize :: Model -> Text -> IO (VU.Vector Token)
  llamaTokenize model txt =
    withModel model $ \model_ptr -> 
      withCString (T.unpack txt) $ \str -> mask_ $
        alloca $ \tokens_ptr_ptr ->
        alloca $ \tokens_sz_ptr -> do
          result <- c_llama_tokenize model_ptr str tokens_ptr_ptr tokens_sz_ptr
          when (result /= 0) $
            pomneticError "Failed to tokenize"

          ntokens <- peek tokens_sz_ptr
          tokens_ptr <- peek tokens_ptr_ptr

          tokens <- VU.generateM (safeFromIntegral ntokens) $ \idx -> do
            val <- peekElemOff tokens_ptr idx
            return $ Token $ safeFromIntegral val

          c_llama_free_tokens tokens_ptr

          return tokens

-- | Converts a token to its text piece. May throw `InvalidToken` if the token
-- is not valid.
--
-- Also note: there is no guarantee that every token has a valid unicode text
-- piece, depending on how it implements tokenization.
--
-- Note: a group of tokens converted to text together can differ from
-- individually converted tokens. Try to convert tokens en-masse.
tokensToText :: Model -> VU.Vector Token -> Text
tokensToText _model tokens | Just token <- VU.find (\(Token t) -> t < 0) tokens = throw $ InvalidToken token
tokensToText model tokens = unsafePerformIO $ do
  ref <- readIORef (hfTokenizer model)
  case ref of
    Nothing -> llamaTokensToText model tokens
    Just hf -> tokensToTextByHF (hfModelID hf) tokens
 where
  llamaTokensToText :: Model -> VU.Vector Token -> IO Text
  -- no bulk conversion in llama.cpp
  llamaTokensToText model tokens = do
    results <- for (VU.toList tokens) $ \token -> llamaTokenToText model token
    return $ mconcat results

  llamaTokenToText :: Model -> Token -> IO Text
  llamaTokenToText model (Token token) = withModel model $ \model_ptr -> do
    vocab_size <- safeFromIntegral <$> c_llama_vocab_size_model model_ptr

    when (token >= vocab_size) $
      throwIO $ InvalidToken (Token token)

    alloca $ \str_ptr ->
     alloca $ \str_len_ptr -> mask_ $ do
      result <- c_llama_token_to_text model_ptr token str_ptr str_len_ptr
      when (result /= 0) $
        pomneticError "Failed to convert token to text"

      str <- peek str_ptr
      str_len <- peek str_len_ptr

      -- TODO: peekCStringLen uses current locale, which is not necessarily UTF-8
      result <- peekCStringLen (str, safeFromIntegral str_len)
      c_llama_free_text str

      return $ T.pack result
