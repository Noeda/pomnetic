{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE MultiWayIf #-}
{-# LANGUAGE LambdaCase #-}

module Pomnetic
  ( newManager
  , PomneticError(..)
  , withSession
  , wholeText
  , addText
  , resetText
  , generateText )
  where

import Control.Concurrent
import Control.Concurrent.STM
import Control.Exception
import Control.Monad
import Control.Monad.IO.Class
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
import Pomnetic.Medium
import System.IO

data Manager = Manager
  { cmModel :: !Model
  , cmContext :: !Context
  , cmAvailableSeqIdxs :: !(TVar IntSet)
  , cmCollectedWork :: !(TVar (SQ.Seq Work)) }

data Work = Predict !SeqID !Int ((BatchItem -> IO Int) -> IO ()) (Context -> Batch -> IO ()) (IO ())
  --                 seq_id sz     lay out the items               -- called on result        reset

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

newManager :: MonadIO m => FilePath -> m Manager
newManager fpath = liftIO $ mask_ $ do
  model <- loadModel fpath
  ctx <- createContext model
  work <- newTVarIO SQ.empty

  tid <- forkIOWithUnmask $ \unmask -> unmask $ do
    result <- try $ worker model ctx work
    case result of
      Left KillSilently -> return ()
      Right () -> error "impossible"

  void $ mkWeakTVar work $ throwTo tid KillSilently

  available_seq_idxs <- newTVarIO $ IS.fromList [0..99]
  return $ Manager model ctx available_seq_idxs work

worker :: Model -> Context -> TVar (SQ.Seq Work) -> IO ()
worker model ctx work_queue = forever $ do
  work_items <- atomically $ do
    seq <- readTVar work_queue
    when (SQ.null seq) retry
    writeTVar work_queue SQ.empty
    return seq

  handleWorkItems model ctx (toList work_items)

handleWorkItems :: Model -> Context -> [Work] -> IO ()
handleWorkItems _model ctx works = do
  let num_items = sum $ fmap workSize works
  if num_items == 0
    then return ()
    else go num_items
 where
  go num_items = do
    batch <- createBatch num_items
    setBatchLength batch num_items

    go2 batch 0 works
    result <- try $ decode ctx batch
    case result of
      Left (PomneticError msg) -> throwIO $ PomneticError msg
      Left TooLongText | length works > 1 -> do
        hPutStrLn stderr "Needing to break up batches..."
        for_ works $ \work ->
          workReset work
        handleWorkItems _model ctx (take (length works `div` 2) works)
        handleWorkItems _model ctx (drop (length works `div` 2) works)
      Left TooLongText -> throwIO TooLongText
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

data Session = Session
  { posRef :: !(IORef Int)
  , nextTokenRef :: !(IORef Int)
  , sessionMu :: !(IORef Float)
  , generatedTokens :: !(IORef (Vector Token))
  , wantedTokens :: !(IORef (Vector Token))
  , logitsTVar :: !(TVar (Maybe Logits))
  , sessionManager :: !Manager
  , sessionSeqIdx :: !Int }

withSession :: MonadIO m => Manager -> (Session -> IO a) -> m a
withSession manager action = liftIO $ withSeqIdx manager $ \seq_idx -> do
  removeTokens (cmContext manager) seq_idx 0 (-1)

  pos_ref <- newIORef 0
  next_token_ref <- newIORef 0

  mu_ref <- newIORef 8.0
  logits_tvar <- newTVarIO Nothing

  let bos = bosToken (cmContext manager)

  gen_ref <- newIORef V.empty
  wanted_ref <- newIORef $ V.singleton bos

  let session = Session { posRef = pos_ref
                        , nextTokenRef = next_token_ref
                        , sessionMu = mu_ref
                        , generatedTokens = gen_ref
                        , wantedTokens = wanted_ref
                        , logitsTVar = logits_tvar
                        , sessionManager = manager
                        , sessionSeqIdx = seq_idx }

  result <- action session

  removeTokens (cmContext manager) seq_idx 0 (-1)

  return result

wholeText :: MonadIO m => Session -> m Text
wholeText session = liftIO $ do
  applyWantedTokens session
  tokens <- readIORef (generatedTokens session)

  let model = cmModel (sessionManager session)
  let txt = mconcat $ fmap (tokenToText model) (V.toList tokens)

  return txt

generateText :: MonadIO m => Session -> Int -> m ()
generateText _ 0 = return ()
generateText session max_tokens = liftIO $ do
  applyWantedTokens session

  let seq_idx = sessionSeqIdx session
      manager = sessionManager session

  logits <- fmap fromJust $ atomically $ readTVar (logitsTVar session)

  mu <- readIORef (sessionMu session)
  (new_token, new_mu) <- sampleMirostat (cmContext $ sessionManager session) logits mu
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

  generateText session (max_tokens-1)

applyWantedTokens :: Session -> IO ()
applyWantedTokens session = do
  gen_vec <- readIORef (generatedTokens session)
  wanted_vec <- readIORef (wantedTokens session)

  if | V.length gen_vec > V.length wanted_vec -> do
         removeTokens ctx seq_idx (V.length wanted_vec-1) (-1)
         writeIORef (generatedTokens session) (V.take (V.length wanted_vec-1) gen_vec)
         writeIORef (posRef session) (V.length wanted_vec-1)
         applyWantedTokens session

     | otherwise -> do
         -- count common prefix length
         len <- commonPrefixLen gen_vec wanted_vec 0
         if len == V.length wanted_vec
           then return ()  -- same prefixes
           else do removeTokens ctx seq_idx len (-1)
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

addText :: MonadIO m => Session -> Text -> m ()
addText session text = liftIO $ do
  let model = cmModel (sessionManager session)
      new_tokens = tokenize model text
  modifyIORef' (wantedTokens session) (<> new_tokens)

resetText :: MonadIO m => Session -> m ()
resetText session = liftIO $ do
  let bos = bosToken (cmContext $ sessionManager session)
  writeIORef (wantedTokens session) (V.singleton bos)
