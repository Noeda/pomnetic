{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Concurrent.Async
import Control.Monad
import Data.Attoparsec.ByteString
import Data.Aeson.Parser
import qualified Data.Text as T
import Pomnetic

main :: IO ()
main = do
  manager <- newManager "zephyr-7b-beta.Q6_K.gguf"
    (setAfterGenWaitMs 500 $
     setStartGenAfterNWaiters 5 defaultManagerSettings)

  forConcurrently_ [1..5] $ \idx ->
    withSession manager $ \session -> do
      let gen20_tokens' = generateConfig 20
          ap = attoparsecBSFilter $ void $ word8 32 >> void json

          gen20_tokens = gen20_tokens' { filters = ap }

      addText session $ T.pack $ "Hi, my name is Jacob and I like the number " <> show idx <> ". Here is some JSON to prove it:"
      generateText session gen20_tokens
      txt <- wholeText session
      print (idx, txt)
      generateText session gen20_tokens
      txt <- wholeText session
      print (idx, txt)
      resetText session

      addText session $ T.pack $ "Hi, my name is Rachel"
      generateText session gen20_tokens
      txt <- wholeText session
      print (idx, txt)
      resetText session


