LLM swarm utility thing
=======================

These are bindings to `llama.cpp` with some focus on making it simpler in
Haskell to write code that runs several instances of AI at once, and doing it
efficiently.

You can write code with dozens of instances of text generation from Haskell
threads and it'll do its best to transparently batch them together and run them
efficiently.

```haskell
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

  -- spawn 50 threads, each with a separate text generation instance
  forConcurrently_ [1..50] $ \idx ->
    withSession manager $ \session -> do
      -- One session = one independent text generation instance

      -- Insert a prompt (session starts with empty prompt)
      addText session $ T.pack $ "Hi, my name is Jacob and I like the number " <> show idx <> " for these reasons:"

      -- Generate 20 tokens
      generateText session 20

      -- Obtain the current text (will include the prompt + the 20 tokens)
      txt <- wholeText session
      print (idx, txt)

      -- Generate 20 more tokens
      generateText session 20
      txt <- wholeText session
      print (idx, txt)

      -- Empty all text from the session (so it's like if you started over)
      resetText session

      addText session $ T.pack $ "Hi, my name is Rachel"
      generateText session 20
      txt <- wholeText session
      print (idx, txt)

      resetText session
```


