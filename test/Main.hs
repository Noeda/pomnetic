{-# LANGUAGE OverloadedStrings #-}

module Main ( main ) where

import Control.Exception
import Control.DeepSeq
import Control.Monad
import Data.Maybe
import Pomnetic
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
-- a valid file and has *some* content.
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
      result <- try $ evaluate $ tokenToText model invalid_token

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
      evaluate $ rnf tokens3_sp
  ]
