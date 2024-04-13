{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE QuasiQuotes #-}

module Main where

import Control.Concurrent.Async
import Control.Monad
import Data.Attoparsec.ByteString
import Data.Aeson.Parser
import Data.Foldable
import Data.IntMap.Strict (IntMap)
import qualified Data.IntMap.Strict as IM
import Data.Text
import qualified Data.Text as T
import qualified Data.Vector.Unboxed as VU
import NeatInterpolation
import Pomnetic
import System.Environment

main :: IO ()
main = do
  args <- getArgs
  case args of
    [model_path] -> run model_path
    _ -> putStrLn "Usage: pomnetic-more-agents-is-all-you-need-experiment <model_path>"

-- I won't type it every time, so I'm defining this two-letter acronym in rest
-- of the code in this file:
--
-- MA = More Agents Is All You Need
--

data SamplingStateMA = SamplingStateMA
  { }
  deriving ( Eq, Ord, Show, Read )

testPrompt :: Text
testPrompt = [trimming|
# transcript_mathematics_formulas_and_answers.txt

62 + 83 = 145
771 - 62 = 709
920 * 3 = 2760
721 * 1128 = 813288
921 * 771 =
|]

type Probability = Double

-- | Given raw logits, returns a map of potential samples, with a simple
-- "minimum 5% probability" thresholding.
--getPotentialSamples :: Logits -> IntMap Probability
--getPotentialSamples logits =
--  fg

run :: FilePath -> IO ()
run model_filepath = do
  manager <- newManager model_filepath
    (setAfterGenWaitMs 500 $
     setStartGenAfterNWaiters 5 defaultManagerSettings)

  withSession manager $ \session -> do
    let model = sessionModel session

    addText session testPrompt
    logits <- nextLogits session
    let l = softmaxLogits $ sortLogits logits
    VU.forM_ l $ \(idx, prob) -> do
      let txt = tokensToText model (VU.singleton idx)
      print (idx, prob, txt)
