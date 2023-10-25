{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Concurrent.Async
import Data.Foldable
import qualified Data.Text as T
import Pomnetic
import qualified Data.Vector as V

main :: IO ()
main = do
  manager <- newManager "llama-2-7b-chat.Q6_K.gguf"

  forConcurrently_ [1..50] $ \idx ->
    withSession manager $ \session -> do
      addText session $ T.pack $ "Hi, my name is Jacob and I like the number " <> show idx <> " for these reasons:"
      generateText session 20
      txt <- wholeText session
      print (idx, txt)
      generateText session 20
      txt <- wholeText session
      print (idx, txt)
      resetText session

      addText session $ T.pack $ "Hi, my name is Rachel"
      generateText session 20
      txt <- wholeText session
      print (idx, txt)
      resetText session


