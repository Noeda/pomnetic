{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Concurrent.Async
import qualified Data.Text as T
import Pomnetic

main :: IO ()
main = do
  manager <- newManager "zephyr-7b-beta.Q6_K.gguf"
    (setAfterGenWaitMs 500 $
     setStartGenAfterNWaiters 5 $
     enableDebugLog defaultManagerSettings)

  forConcurrently_ [1..5] $ \idx ->
    withSession manager $ \session -> do
      let gen20_tokens = generateConfig 20

      addText session $ T.pack $ "Hi, my name is Jacob and I like the number " <> show idx <> " for these reasons:"
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


