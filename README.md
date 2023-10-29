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

main :: IO ()
main = do
  -- Create a manager, with settings as such:
  -- To start text generation, wait at most 500ms for batch queues to become
  -- full, OR until at least 5 threads are enqueued waiting for text
  -- processing. (5 because we are going to spawn 5 threads, so we get about
  -- the best batching we can).
  manager <- newManager "zephyr-7b-beta.Q6_K.gguf"
    (setAfterGenWaitMs 500 $
     setStartGenAfterNWaiters 5 defaultManagerSettings)

  -- Spawn 5 threads, each generating text independently.
  forConcurrently_ [1..5] $ \idx ->
    withSession manager $ \session -> do
      -- Default generation config using mirostat; generate 20 tokens at a
      -- time.
      let gen20_tokens = generateConfig 20

      -- Add a prompt to the session. Sessions start with an empty text.
      addText session $ T.pack $ "Hi, my name is Jacob and I like the number " <> show idx <> " for these reasons:"
      generateText session gen20_tokens

      -- Get the current text in the session (includes prompt + the 20
      generated tokens)
      txt <- wholeText session
      print (idx, txt)

      -- Do it again
      generateText session gen20_tokens
      txt <- wholeText session
      print (idx, txt)

      -- Empty out the session (no text will remain, it's like starting over)
      resetText session

      -- Start generating something else.
      addText session $ T.pack $ "Hi, my name is Rachel"
      generateText session gen20_tokens
      txt <- wholeText session
      print (idx, txt)

      resetText session

      -- etc.
```


