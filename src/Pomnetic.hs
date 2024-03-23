-- | High-level bindings to llama.cpp
--
-- This is a higher level API to llama.cpp text generation.
--
-- There is a more low-level API in Pomnetic.Medium, and this module is built
-- on top of that layer.
--
-- Designed to be easy to use from multiple Haskell threads; if you use the
-- same `Manager` from multiple threads, this will, behind the scenes, try to
-- batch your generation tasks in efficient bundles.
--

{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiWayIf #-}
{-# LANGUAGE LambdaCase #-}

module Pomnetic
  ( newManager
  , newManagerFromModel
  , Model()
  , loadModel
  , loadModelWithHFTokenizer
  , hfTokenize
  , hfTokenizeEmpty
  , Manager()
  , ManagerSettings()
  , defaultManagerSettings
  , enableDebugLog
  , setMaxContextLength
  , setAfterGenWaitMs
  , afterGenWaitMs
  , startGenAfterNWaiters
  , setStartGenAfterNWaiters
  , PomneticError(..)
  , withSession
  , Session()
  , sessionContext
  , sessionModel
  , wholeText
  , wholeTokens
  , textFrom
  , addText
  , addTokens
  , resetText
  , generateText
  -- * Generation configuration
  , GenerateConfig(..)
  , generateConfig
  , Sampler(..)
  -- ** Raw logits
  , nextLogits
  , Logits
  , tokensToText
  , intToToken
  , tokenToInt
  , vocabularySize
  -- ** Filters
  , Filters()
  , andFilters
  , orFilters
  , regexFilter
  , attoparsecFilter
  , attoparsecBSFilter
  -- ** Mirostat
  , MirostatConfig(..)
  , mirostatConfig
  , mirostat4
  , MirostatMu )
  where

import Control.Concurrent
import Control.Concurrent.STM
import Control.Exception
import Control.Monad
import Control.Monad.IO.Class
import Control.Monad.State.Strict
import Data.Foldable
import Data.Data
import Data.IORef
import Data.IntSet ( IntSet )
import qualified Data.IntSet as IS
import Data.Maybe ( fromJust )
import qualified Data.Sequence as SQ
import Data.Text ( Text )
import Data.Vector ( Vector )
import qualified Data.Vector as V
import GHC.Generics
import Pomnetic.HuggingFaceTokenizers
import Pomnetic.Medium
import System.Clock
import System.IO
import System.Timeout

data Manager = Manager
  { cmModel :: !Model
  , cmContext :: !Context
  , cmAvailableSeqIdxs :: !(TVar IntSet)
  , cmCollectedWork :: !(TVar (SQ.Seq Work)) }

data Work = Predict !SeqID !Int ((BatchItem -> IO Int) -> IO ()) (Context -> Batch -> IO ()) (IO ())
  --                 seq_id sz     lay out the items               -- called on result        reset

workSeqID :: Work -> SeqID
workSeqID (Predict seq_id _ _ _ _) = seq_id

workSize :: Work -> Int
workSize (Predict _ sz _ _ _) = sz

workReset :: Work -> IO ()
workReset (Predict _ _ _ _ reset) = reset

workLayer :: Work -> (BatchItem -> IO Int) -> IO ()
workLayer (Predict _ _ lay _ _) = lay

workResulter :: Work -> Context -> Batch -> IO ()
workResulter (Predict _ _ _ res _) = res

data KillSilently = KillSilently
  deriving ( Eq, Ord, Show, Typeable )

instance Exception KillSilently

type UseDebugLog = Bool

-- | Settings for a manager.
data ManagerSettings = ManagerSettings
  { debugLog :: !UseDebugLog
  , afterGenWaitMs :: !Integer
  , startGenAfterNWaiters :: !Integer
  , maxContextLength :: !(Maybe Int) }
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic )

defaultManagerSettings :: ManagerSettings
defaultManagerSettings = ManagerSettings
  { debugLog = False
  , startGenAfterNWaiters = 0
  , afterGenWaitMs = 0
  , maxContextLength = Nothing }

-- | Sets maximum context length for any loaded model.
--
-- Normally, the maximum context length is determined by the model itself, but
-- you can override it with this function.
--
-- There are models out there with very long context lengths.
--
-- Default is no limit.
setMaxContextLength :: Int -> ManagerSettings -> ManagerSettings
setMaxContextLength n settings = settings { maxContextLength = Just n }

-- | Enables debug log. This will make the manager write to stderr about
-- everything that happens. Noisy.
enableDebugLog :: ManagerSettings -> ManagerSettings
enableDebugLog settings = settings { debugLog = True }

-- | Sets a wait time in milliseconds after genereating some tokens.
--
-- By default this is 0. Why would you want to add a delay? The reason is that
-- if you have multiple threads, all vying for their turn to generate text,
-- instantly picking up new work and processing it will most likely not get
-- every thread on board, and some of them will wait (normally they'd be
-- batched; which is more efficient). A tiny delay (e.g. 5ms) is often already
-- enough to mitigate this.
--
-- If `startGenAfterNWaiters` is also set, then that can trigger generation
-- earlier.
setAfterGenWaitMs :: Integer -> ManagerSettings -> ManagerSettings
setAfterGenWaitMs ms settings = settings { afterGenWaitMs = ms }

-- | Sets a threshold that if reached, the manager will start processing tokens.
--
-- This is used with `setAfterGenWaitMs`. For example, if you know that you
-- have 5 active threads trying to generate text, you can set this threshold to
-- 5 and as soon as those 5 threads have submitted a request to generate text,
-- the processing begins, rather than waiting until the time limit set in
-- `setAfterGenWaitMs`.
--
-- Has no effect if `setAfterGenWaitMs` is not also set.
--
-- The default is 0, which will turn this feature off; making the generator
-- always wait until `setAfterGenWaitMs`.
setStartGenAfterNWaiters :: Integer -> ManagerSettings -> ManagerSettings
setStartGenAfterNWaiters n settings = settings { startGenAfterNWaiters = n }

-- | Creates a new manager, from a `Model`. Otherwise same as `newManager`.
newManagerFromModel :: MonadIO m
                    => Model
                    -> ManagerSettings
                    -> m Manager
newManagerFromModel model manager_settings = liftIO $ mask_ $ do
  settings' <- makeSensibleDefaultContextSettings model

  let settings = settings' { contextSettingsMaxTokens = case maxContextLength manager_settings of
                                                         Nothing -> contextSettingsMaxTokens settings'
                                                         Just n -> min (contextSettingsMaxTokens settings') n }

  ctx <- createContext model settings
  work <- newTVarIO SQ.empty

  tid <- forkIOWithUnmask $ \unmask -> unmask $ do
    result <- try $ worker model ctx work
                      (debugLog manager_settings)
                      (afterGenWaitMs manager_settings)
                      (startGenAfterNWaiters manager_settings)
    case result of
      Left KillSilently -> return ()
      Right () -> error "impossible"

  void $ mkWeakTVar work $ throwTo tid KillSilently

  available_seq_idxs <- newTVarIO $ IS.fromList [0..99]
  return $ Manager model ctx available_seq_idxs work

-- | Creates a new manager.
--
-- One manager handles one or more sessions generating text, batching their
-- text generation in an efficient way behind the scenes if they are generating
-- text at the same time from multiple threads.
--
-- If your program is likely going to use multiple managers, you may want to
-- use `loadModel` and `newManagerFromModel` instead, so that all managers can
-- share the same model.
newManager :: MonadIO m
           => FilePath
           -> ManagerSettings
           -> m Manager
newManager fpath manager_settings = do
  model <- loadModel fpath
  newManagerFromModel model manager_settings

worker :: Model -> Context -> TVar (SQ.Seq Work) -> UseDebugLog -> Integer -> Integer -> IO ()
worker model
       ctx
       work_queue
       use_debug_log
       after_gen_wait_ms
       start_gen_after_n_waiters = forever $ do
  work_items <- atomically $ do
    seq <- readTVar work_queue
    when (SQ.null seq) retry
    writeTVar work_queue SQ.empty
    return seq

  handleWorkItems model ctx (toList work_items) use_debug_log
  unless (null work_items) $
    when (after_gen_wait_ms > 0) $
      if start_gen_after_n_waiters == 0
        then threadDelay $ fromIntegral after_gen_wait_ms * 1000
        else void $ timeout (fromIntegral after_gen_wait_ms * 1000) $ atomically $ do
            works <- readTVar work_queue
            unless (SQ.length works >= fromIntegral start_gen_after_n_waiters) retry

handleWorkItems :: Model -> Context -> [Work] -> UseDebugLog -> IO ()
handleWorkItems _model ctx works use_debug_log = do
  let num_items = sum $ fmap workSize works
  if num_items == 0
    then return ()
    else go num_items
 where
  go num_items = do
    batch <- createBatch num_items
    setBatchLength batch num_items

    go2 batch 0 works
    start <- getTime Monotonic
    when use_debug_log $ do
      -- Count different number of sequences
      let n_seq_ids = IS.size $ flip execState IS.empty $ for_ works $ \work ->
                        modify $ IS.insert (workSeqID work)
      hPutStrLn stderr $ "processBatch called with length=" <> show num_items <> " and n_seq_ids=" <> show n_seq_ids
    result <- try $ processBatch ctx batch
    end <- getTime Monotonic
    when use_debug_log $ do
      let nanosecs = toNanoSecs $ diffTimeSpec end start
          per_item = nanosecs `div` (fromIntegral num_items)
      hPutStrLn stderr $ "processBatch finished: " <> show (fromIntegral nanosecs / 1000000000 :: Double) <> " seconds (" <> show (fromIntegral per_item / 1000000000 :: Double) <> " per item)"
    case result of
      Left (PomneticError msg) -> throwIO $ PomneticError msg
      Left TooLongText | length works > 1 -> do
        when use_debug_log $
          hPutStrLn stderr $ "Received TooLongText exception, batch size " <> show num_items <> " will try again by splitting into two."
        for_ works $ \work ->
          workReset work
        handleWorkItems _model ctx (take (length works `div` 2) works) use_debug_log
        handleWorkItems _model ctx (drop (length works `div` 2) works) use_debug_log
      Left TooLongText -> throwIO TooLongText
      Left other -> throwIO other
      Right () -> for_ works $ \work ->
        workResulter work ctx batch
   where
    go2 _batch _batch_idx [] = return ()
    go2 batch batch_idx (work:rest) = do
      for_ [0..workSize work-1] $ \work_idx ->
        workLayer work (\item -> do
                         setBatchItem batch item (batch_idx+work_idx)
                         return (batch_idx+work_idx))
      go2 batch (batch_idx+workSize work) rest

withSeqIdx :: Manager -> (SeqID -> IO a) -> IO a
withSeqIdx manager action = mask $ \restore -> do
  seq_idx <- atomically $ do
    seq_idxs <- readTVar (cmAvailableSeqIdxs manager)
    case IS.minView seq_idxs of
      Nothing -> retry
      Just (seq_idx, rest) -> do
        writeTVar (cmAvailableSeqIdxs manager) rest
        return seq_idx

  finally (restore $ action seq_idx) $ do
    atomically $ modifyTVar (cmAvailableSeqIdxs manager) (IS.insert seq_idx)

-- | One session of text generation.
data Session = Session
  { posRef :: !(IORef Int)
  , sessionMu :: !(IORef Float)
  , generatedTokens :: !(IORef (Vector Token))
  , wantedTokens :: !(IORef (Vector Token))
  , logitsTVar :: !(TVar (Maybe Logits))
  , sessionManager :: !Manager
  , sessionSeqIdx :: !Int }

sessionContext :: Session -> Context
sessionContext = cmContext . sessionManager

sessionModel :: Session -> Model
sessionModel = cmModel . sessionManager

-- | Creates a new session, runs code inside the session, and then cleans up
-- the session.
--
-- Bad things may happen if you keep the `Session` value after `withSession` is
-- over. Only use it inside its own `withSession`.
withSession :: MonadIO m => Manager -> (Session -> IO a) -> m a
withSession manager action = liftIO $ withSeqIdx manager $ \seq_idx -> do
  forgetTokens (cmContext manager) seq_idx 0 (-1)

  pos_ref <- newIORef 0

  mu_ref <- newIORef 8.0
  logits_tvar <- newTVarIO Nothing

  let bos = bosToken (cmContext manager)

  gen_ref <- newIORef V.empty
  wanted_ref <- newIORef $ V.singleton bos

  let session = Session { posRef = pos_ref
                        , sessionMu = mu_ref
                        , generatedTokens = gen_ref
                        , wantedTokens = wanted_ref
                        , logitsTVar = logits_tvar
                        , sessionManager = manager
                        , sessionSeqIdx = seq_idx }

  result <- action session

  forgetTokens (cmContext manager) seq_idx 0 (-1)

  return result

-- | Returns the current text in the session.
wholeText :: MonadIO m => Session -> m Text
wholeText session = liftIO $ do
  applyWantedTokens session
  tokens <- readIORef (generatedTokens session)

  let model = cmModel (sessionManager session)
  let txt = tokensToText model tokens

  return txt

-- | Returns the current text in the session, in the form of tokens.
--
-- Does not include the BOS token.
wholeTokens :: MonadIO m => Session -> m (Vector Token)
wholeTokens session = liftIO $ do
  applyWantedTokens session
  result <- readIORef (generatedTokens session)
  return $ V.tail result

-- | Returns text starting from a certain index (in tokens).
textFrom :: MonadIO m => Session -> Int -> m Text
textFrom session start = liftIO $ do
  applyWantedTokens session
  tokens <- readIORef (generatedTokens session)

  let model = cmModel (sessionManager session)
  let txt = tokensToText model (V.drop start tokens)

  return txt

data GenerateConfig = GenerateConfig
  { numTokens :: !Int
  , filters :: !Filters
  , sampler :: !Sampler }
  deriving ( Typeable, Generic )

data Sampler
  = Mirostat !MirostatConfig
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic )

-- | Generation config with mirostat4 as the sampler and no filters.
--
-- Generates N tokens.
generateConfig :: Int -> GenerateConfig
generateConfig ntokens = GenerateConfig
  { numTokens = ntokens
  , filters = mempty
  , sampler = Mirostat mirostat4 }

-- | Generates N amount of tokens to the session.
generateText :: MonadIO m => Session -> GenerateConfig -> m ()
generateText session config = liftIO $ do
  applyWantedTokens session
  n_tokens_generated <- fmap V.length $ readIORef (generatedTokens session)
  generateText2 session config n_tokens_generated

-- | Returns logits for what would be the next token.
nextLogits :: MonadIO m => Session -> m Logits
nextLogits session = liftIO $ do
  applyWantedTokens session
  logits <- fmap fromJust $ atomically $ readTVar (logitsTVar session)
  return logits

generateText2 :: Session -> GenerateConfig -> Int -> IO ()
generateText2 _ config _ | numTokens config == 0 = return ()
generateText2 session config n_tokens_generated = liftIO $ do
  applyWantedTokens session

  let seq_idx = sessionSeqIdx session
      manager = sessionManager session

  regex_filter_text <- textFrom session n_tokens_generated

  logits <- fmap fromJust $ atomically $ readTVar (logitsTVar session)

  mu <- readIORef (sessionMu session)
  (new_token, new_mu) <- case sampler config of
    Mirostat mirostat_config ->
      sampleMirostat (cmContext $ sessionManager session)
                     logits
                     mu
                     (filters config)
                     mirostat_config
                     regex_filter_text

  writeIORef (sessionMu session) new_mu

  modifyIORef' (generatedTokens session) $ \vec -> vec <> V.singleton new_token
  modifyIORef' (wantedTokens session) $ \vec -> vec <> V.singleton new_token

  original_pos <- readIORef (posRef session)
  gen_vec <- readIORef (generatedTokens session)

  target_idx <- newIORef 0

  atomically $ writeTVar (logitsTVar session) Nothing

  let reset = writeIORef (posRef session) original_pos

      layer set_item = do pos <- readIORef (posRef session)
                          writeIORef (posRef session) (pos+1)

                          idx <- set_item (BatchItem {
                                       token = V.last gen_vec,
                                       position = pos,
                                       sequenceId = sessionSeqIdx session,
                                       logits = True })
                          writeIORef target_idx idx

      obtain_logits ctx _batch = do target <- readIORef target_idx
                                    logits <- getLogits ctx target
                                    atomically $ writeTVar (logitsTVar session) (Just logits)

  atomically $ do
    modifyTVar (cmCollectedWork manager) $ \seq ->
      seq SQ.|> Predict seq_idx 1 layer obtain_logits reset

  atomically $ readTVar (logitsTVar session) >>= \case
    Nothing -> retry
    Just _logits -> return ()

  generateText2 session (config { numTokens = numTokens config - 1 }) n_tokens_generated

applyWantedTokens :: Session -> IO ()
applyWantedTokens session = do
  gen_vec <- readIORef (generatedTokens session)
  wanted_vec <- readIORef (wantedTokens session)

  if | V.length gen_vec > V.length wanted_vec -> do
         forgetTokens ctx seq_idx (V.length wanted_vec-1) (-1)
         writeIORef (generatedTokens session) (V.take (V.length wanted_vec-1) gen_vec)
         writeIORef (posRef session) (V.length wanted_vec-1)
         applyWantedTokens session

     | otherwise -> do
         -- count common prefix length
         len <- commonPrefixLen gen_vec wanted_vec 0
         if len == V.length wanted_vec
           then return ()  -- same prefixes
           else do forgetTokens ctx seq_idx len (-1)
                   writeIORef (generatedTokens session) (V.take len gen_vec)
                   decodeWantedTokens (V.drop len wanted_vec)
 where
  commonPrefixLen gen_vec wanted_vec idx =
     if V.length gen_vec > idx && V.length wanted_vec > idx &&
        gen_vec V.! idx == wanted_vec V.! idx
       then commonPrefixLen gen_vec wanted_vec (idx+1)
       else return idx

  seq_idx = sessionSeqIdx session
  ctx = cmContext manager
  manager = sessionManager session

  -- have llama.cpp process the text without predicting next token
  decodeWantedTokens wanted_vec = do
    gen_vec <- readIORef (generatedTokens session)

    let reset = writeIORef (posRef session) (V.length gen_vec)
    reset

    target_ref <- newIORef 0
    atomically $ writeTVar (logitsTVar session) Nothing

    let layer set_item = do pos <- readIORef (posRef session)
                            writeIORef (posRef session) (pos+1)

                            let use_logits = pos - V.length gen_vec == V.length wanted_vec - 1

                            idx <- set_item (BatchItem { token = wanted_vec V.! (pos - V.length gen_vec)
                                                       , position = pos
                                                       , sequenceId = seq_idx
                                                       , logits = use_logits })

                            writeIORef target_ref idx

        obtain_logits ctx _batch = do target <- readIORef target_ref
                                      logits <- getLogits ctx target
                                      atomically $ writeTVar (logitsTVar session) (Just logits)

    atomically $ do
      modifyTVar (cmCollectedWork manager) $ \seq ->
        seq SQ.|> Predict seq_idx (V.length wanted_vec) layer obtain_logits reset
    atomically $ readTVar (logitsTVar session) >>= \case
      Nothing -> retry
      Just _logits -> return ()

    w <- readIORef $ wantedTokens session
    writeIORef (generatedTokens session) w

-- | Adds text to the end of the current session.
addText :: MonadIO m => Session -> Text -> m ()
addText session text = liftIO $ do
  let model = cmModel (sessionManager session)
      new_tokens = tokenize model text
  modifyIORef' (wantedTokens session) (<> new_tokens)

-- | Adds tokens to the end of the current session.
addTokens :: MonadIO m => Session -> Vector Token -> m ()
addTokens session tokens = liftIO $ do
  modifyIORef' (wantedTokens session) (<> tokens)

-- | Erases all text from the session.
resetText :: MonadIO m => Session -> m ()
resetText session = liftIO $ do
  let bos = bosToken (cmContext $ sessionManager session)
  writeIORef (wantedTokens session) (V.singleton bos)

-- | Mirostat sampling with tau = 4 and eta = 0.05
mirostat4 :: MirostatConfig
mirostat4 = MirostatConfig
  { mirostatTau = 4.0
  , mirostatEta = 0.05
  }
