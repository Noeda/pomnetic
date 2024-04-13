{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE QuasiQuotes #-}

module Main where

import Control.Concurrent.Async
import Control.Monad
import Data.Attoparsec.ByteString
import Data.Aeson.Parser
import Data.Foldable
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as M
import Data.Text
import qualified Data.Text as T
import Data.Traversable
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

data MAConfig = MAConfig
  { numAgents :: !Int
  , sampleLength :: !Int }

generateMAText :: Session -> MAConfig -> IO ()
generateMAText session maconfig = do
  all_tokens <- wholeTokens session

  completions <- for [0..numAgents maconfig-1] $ \agent_id -> do
    putStrLn $ "Agent " ++ show agent_id ++ " is generating text..."

    resetText session
    addTokens session all_tokens

    generateText session (generateConfig (sampleLength maconfig))
    completion <- VU.drop (VU.length all_tokens) <$> wholeTokens session
    return completion

  print completions

type Probability = Float

-- | Given raw logits, returns a map of potential samples, with a simple
-- "minimum 5% probability" thresholding.
getPotentialSamples :: Logits -> Map Token Probability
getPotentialSamples logits =
  let sorted_logits = sortLogits logits
      softmaxed_logits = softmaxLogits sorted_logits
      filtered_logits = VU.filter (\(_, prob) -> prob > 0.05) softmaxed_logits
      normalized_logits = normalizeLogits filtered_logits
   in M.fromList $ VU.toList normalized_logits

run :: FilePath -> IO ()
run model_filepath = do
  manager <- newManager model_filepath defaultManagerSettings

  withSession manager $ \session -> do
    let model = sessionModel session

    addText session testPrompt
    generateMAText session (MAConfig 20 100)
