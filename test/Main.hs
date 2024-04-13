{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE MultiWayIf #-}

module Main ( main ) where

import Control.Exception
import Control.DeepSeq
import Control.Monad
import Data.Foldable
import Data.IORef
import Data.Maybe
import qualified Data.Vector.Unboxed as VU
import Pomnetic
import Pomnetic.Medium
import Pomnetic.HuggingFaceTokenizers

import Test.Tasty
import Test.Tasty.HUnit

-- Possibly at some point, we can include a test .gguf
-- in the repository. Meanwhile, if you get test error
-- , get some test.gguf in your local directory.
--
-- As of writing this test for the first time, I got
-- phi-2 model from microsoft and renamed it to test.gguf.
--
-- The tests in this file don't depend on the model being good, just that it's
-- a valid file and is at least a little bit trained to give the correct output
-- for the most predictable text sequences.
testModel :: FilePath
testModel = "test.gguf"

main :: IO ()
main = defaultMain tests

tests :: TestTree
tests = testGroup "Pomnetic" [
    testCase "newManager works" $ do
      manager <- newManager testModel defaultManagerSettings
      void $ evaluate manager,

    testCase "newManagerFromModel works" $ do
      model <- loadModel testModel

      manager1 <- newManagerFromModel model defaultManagerSettings
      manager2 <- newManagerFromModel model defaultManagerSettings

      void $ evaluate manager1
      void $ evaluate manager2,

    testCase "Invalid token conversion throws InvalidToken" $ do
      model <- loadModel testModel
      let invalid_token = intToToken (-123)
      result <- try $ evaluate $ tokensToText model (VU.singleton invalid_token)

      case result of
        Left (InvalidToken token) | token == invalid_token -> return ()
        _ -> assertFailure $ "Expected InvalidToken exception, got " <> show result,

    testCase "nextLogits works on an empty context" $ do
      manager <- newManager testModel defaultManagerSettings
      withSession manager $ \session -> do
        logits <- nextLogits session
        void $ evaluate $ rnf logits,

    testCase "nextLogits is deterministic" $ do
      manager <- newManager testModel defaultManagerSettings
      logits1 <- withSession manager $ \session -> do
        addText session "Hello, world! This is"
        logits1 <- nextLogits session
        logits2 <- nextLogits session
        logits3 <- nextLogits session

        addText session "x"
        logits4 <- nextLogits session

        assertBool "logits1 == logits2" $ logits1 == logits2
        assertBool "logits2 == logits3" $ logits2 == logits3
        assertBool "logits1 /= logits4" $ logits1 /= logits4

        return logits1

      logits2 <- withSession manager $ \session -> do
        addText session "Hello, world! This is"
        nextLogits session

      assertBool "logits1 == logits2" $ logits1 == logits2,

    testCase "HF tokenizer can be used" $ do
      -- smallest HF model I found on huggingface
      tokens1 <- tokenizeByHF $ (hfTokenize "prajjwal1/bert-tiny" "Hello, world!") { addSpecialTokens = True }
      tokens1_sp <- tokenizeByHF $ (hfTokenize "prajjwal1/bert-tiny" "Hello, world!") { addSpecialTokens = False }
      tokens2 <- tokenizeByHF $ (hfTokenize "prajjwal1/bert-tiny" "12345") { addSpecialTokens = True }
      tokens2_sp <- tokenizeByHF $ (hfTokenize "prajjwal1/bert-tiny" "12345") { addSpecialTokens = False }
      tokens3 <- tokenizeByHF $ (hfTokenize "prajjwal1/bert-tiny" "") { addSpecialTokens = True }
      tokens3_sp <- tokenizeByHF $ (hfTokenize "prajjwal1/bert-tiny" "") { addSpecialTokens = False }

      bos_token <- bosTokenByHF "prajjwal1/bert-tiny"
      eos_token <- eosTokenByHF "prajjwal1/bert-tiny"
      sep_token <- sepTokenByHF "prajjwal1/bert-tiny"

      -- for prajjwal1/bert-tiny, there are no bos or eos tokens
      -- but it does have a sep token
      assertBool "bos_token is Nothing" $ isNothing bos_token
      assertBool "eos_token is Nothing" $ isNothing eos_token
      assertBool "sep_token is 102" $ sep_token == Just (intToToken 102)

      evaluate $ rnf tokens1
      evaluate $ rnf tokens2
      evaluate $ rnf tokens3
      evaluate $ rnf tokens1_sp
      evaluate $ rnf tokens2_sp
      evaluate $ rnf tokens3_sp,

    testCase "Long context with small batch" $ do
      model <- loadModel testModel
      ctx_settings' <- makeSensibleDefaultContextSettings model
      let ctx_settings = ctx_settings' { contextSettingsMaxTokens = 3000
                                       , contextSettingsBatchSize = 13 }
      ctx <- createContext model ctx_settings

      let token_for_a_vec = tokenize model " a"
          token_for_b_vec = tokenize model " b"
          token_for_c_vec = tokenize model " c"

      assertEqual "the ' a' token is just one token" 1 (VU.length token_for_a_vec)
      assertEqual "the ' b' token is just one token" 1 (VU.length token_for_a_vec)
      assertEqual "the ' c' token is just one token" 1 (VU.length token_for_a_vec)
      let token_for_a = VU.head token_for_a_vec
          token_for_b = VU.head token_for_b_vec
          token_for_c = VU.head token_for_c_vec

      batch <- createBatch 1013

      -- makes a pattern abcabcabcabc etc.
      for_ [0..999] $ \idx -> do
        let do_logits = idx == 999
            tok_id = if | idx `mod` 3 == 0 -> token_for_a
                        | idx `mod` 3 == 1 -> token_for_b
                        | otherwise -> token_for_c
        setBatchItem batch (BatchItem { token = tok_id, position = idx, sequenceId = 0, logits = do_logits }) idx
      setBatchLength batch 1000

      processBatch ctx batch

      logits <- getLogits batch 999

      let slogits = VU.take 50 $ sortLogits logits

      -- one of the " a" or " b" or " c" should be near the top
      found <- newIORef False
      VU.forM_ slogits $ \(token_id, _) ->
        when (token_id == token_for_a || token_id == token_for_b || token_id == token_for_c) $
          writeIORef found True

      found' <- readIORef found
      assertBool "one of the tokens should be near the top" found'
  ]
