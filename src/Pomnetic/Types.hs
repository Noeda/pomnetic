{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveGeneric #-}

module Pomnetic.Types
  ( Token(..) )
  where

import Data.Data
import Data.Int
import Foreign.Storable
import GHC.Generics

-- Match with llama.cpp, where it is int32_t
newtype Token = Token Int32
  deriving ( Eq, Ord, Show, Data, Typeable, Storable, Generic )
