{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE QuasiQuotes #-}

module Main where

import Control.Monad
import Data.Foldable
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as M
import Data.Ord
import Data.Text
import qualified Data.Text as T
import Data.Traversable
import qualified Data.Vector.Unboxed as VU
import NeatInterpolation
import Pomnetic
import System.Environment
import System.Random

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
--- SCP-049-J ---

Tags: joke, sapient, biohazard, hostile, euclid, plague-doctor

Special Containment Procedures: SCP-049-J is contained at Site-19 Site-101 Site-17 Site-81 Site-13 Special Restricted High Security Top Secret MK-Ultra Area Region Landmass 101.5 WFML near Richmond, VA. SCP-049-J is permitted to leave its holding cell only under supervision of two (2) (II) (两1) (dos) guards armed with AR-15 rifles and stun batons.

Due to olfactory concerns for staff assigned to SCP-049-J, the entity is no longer allowed to remove its mask.

Description: SCP-049-J is a humanoid entity wearing the period appropriate garb of a medieval plague doctor. Further analysis of SCP-049-J has revealed that under its robes, the entity is composed mostly of moss, wads of tissue, and other, smaller plague doctor masks. It is generally compliant with Foundation staff, but will sometimes lie and occasionally sweat profusely for no reason whatsoever.

During SCP-049-J’s time in Foundation custody, it has continually claimed to be a powerful magical doctor wizard, capable of “curing” that which “ails mankind”. To date, it has been unable to cure literally anything, and typically only exacerbates conditions considerably.

While this alone would not be enough for the Foundation to hold SCP-049-J indefinitely as an anomalous entity, it has also proven capable of somehow always evading capture and escaping from Foundation sites after its true lack of capabilities are revealed. Because of this, and because of staff’s unwavering curiosity as to whether it has any of the self-proclaimed magical healing abilities it describes, SCP-049-J is to be housed and treated as an anomalous entity.

Addendum 049-J.1: Interview

    [BEGIN LOG]

    Dr. Baker: Hello SCP-049-J, welcome to-

    SCP-049-J: I am a doctor.

    Dr. Baker: -uh, yes, I’m aware. We’re just doing this as a-

    SCP-049-J: I have the cure.

    Dr. Baker: (Pauses) …yes, well, we’ll get to that. First off, can you tell me your name?

    SCP-049-J: Yes hmm quite very well I have the cure good sir indubitably yes I am a doctor.

    Dr. Baker: …what?

    SCP-049-J: Bring me to the patient, I will heal them. (Gestures with pointed doctor stick)

    Dr. Baker: Jesus, watch- fuck, watch where you’re swinging that.

    SCP-049-J: I am the cure.

    Dr. Baker: What in the world are you- ohh, I get it. You’re sort of a moron, aren’t you?

    SCP-049-J: Hmm, well quite yes you see it is the pestilence that ails this world.

    Dr. Baker: Yeah, that’s not a- nevermind. Hey, let's roll in the patient.

    *Assistants wheel in D-2569*

    Dr. Baker: Okay, so this gentleman here has a cold. It’s been going around, everyone gets it at least once a year, it’s no big deal. It’s just a common cold.

    SCP-049-J:
|]

data MAConfig = MAConfig
  { numAgents :: !Int
  , sampleLength :: !Int }

generateMAToken :: Session -> MAConfig -> IO ()
generateMAToken session maconfig = do
  all_tokens <- wholeTokens session
  completions <- go all_tokens (numAgents maconfig) (sampleLength maconfig)

  let chosen = similarityFromGroup completions

  resetText session
  addTokens session all_tokens
  addTokens session (VU.singleton (VU.head $ VU.drop (VU.length all_tokens) chosen))
 where
  go :: VU.Vector Token -> Int -> Int -> IO [VU.Vector Token]
  go all_tokens nagents nsamples = do
    next_logits' <- nextLogits session
    let next_logits = getPotentialSamples next_logits'
    go2 all_tokens nagents next_logits nsamples

  go2 :: VU.Vector Token -> Int -> Map Token Probability -> Int -> IO [VU.Vector Token]
  go2 all_tokens nagents next_logits nsamples = do
    assigned_logit <- M.fromListWith (+) <$> for [0..nagents-1] (\_agent_id -> (,) <$> sampleLogitFrom next_logits <*> pure (1 :: Int))
    results <- for (M.assocs assigned_logit) $ \(token_id, nagents_to_assign) -> do
      if nsamples == 1
        then return $ Prelude.replicate nagents_to_assign (all_tokens <> VU.singleton token_id)
        else do resetText session
                addTokens session all_tokens
                addTokens session $ VU.singleton token_id
                go (all_tokens <> VU.singleton token_id) nagents_to_assign (nsamples - 1)
    return $ mconcat results

type Probability = Float

similarityFromGroup :: [VU.Vector Token] -> VU.Vector Token
similarityFromGroup [] = error "impossible"
similarityFromGroup [x] = x
similarityFromGroup xs =
  let similarities = [ (x, sum [ similarity x y | y <- xs ]) | x <- xs ]
      (best, _) = maximumBy (comparing snd) similarities
   in best

similarity :: VU.Vector Token -> VU.Vector Token -> Probability
similarity tokens1 tokens2 | VU.null tokens1 || VU.null tokens2 = 0
similarity tokens1 tokens2 =
  -- occurrence count: how many tokens the two sequces have in common
  let token_counts1 = M.fromListWith (+) $ VU.toList $ VU.map (,1) tokens1
      token_counts2 = M.fromListWith (+) $ VU.toList $ VU.map (,1) tokens2
      common_tokens = M.intersectionWith min token_counts1 token_counts2
      common_tokens_count = sum common_tokens
      similarity = fromIntegral common_tokens_count /
                   fromIntegral (VU.length tokens1 + VU.length tokens2 - common_tokens_count)
   in similarity

-- | Given raw logits, returns a map of potential samples, with a simple
-- "minimum 5% probability" thresholding.
getPotentialSamples :: Logits -> Map Token Probability
getPotentialSamples logits =
  let sorted_logits = sortLogits logits
      softmaxed_logits = normalizeLogits $ VU.take 50 $ softmaxLogits sorted_logits
      filtered_logits = VU.filter (\(_, prob) -> prob > 0.01) softmaxed_logits
   in M.fromList $ VU.toList $ normalizeLogits $ filtered_logits

sampleLogitFrom :: Map Token Probability -> IO Token
sampleLogitFrom logits = do
  r <- randomRIO (0.0, 1.0)
  go (M.assocs logits) r 0.0
 where
  go ((token_id, prob):rest) r accum =
    if r < accum + prob
      then return token_id
      else go rest r (accum + prob)
  go [] _ _ = error "impossible"

run :: FilePath -> IO ()
run model_filepath = do
  manager <- newManager model_filepath defaultManagerSettings

  withSession manager $ \session -> do
    addText session testPrompt
    replicateM_ 1000 $ do
      generateMAToken session (MAConfig 10 20)
      txt <- wholeText session
      putStrLn $ "Text so far:"
      putStrLn $ T.unpack txt
