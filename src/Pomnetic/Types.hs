{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveGeneric #-}

module Pomnetic.Types
  ( Token(..)
  , tokenToInt
  , intToToken
  , HFTokenize(..)
  )
  where

import Control.DeepSeq
import Data.Data
import Data.Int
import Data.Text ( Text )
import Foreign.Storable
import GHC.Generics

-- Match with llama.cpp, where it is int32_t
newtype Token = Token Int32
  deriving ( Eq, Ord, Show, Data, Typeable, Generic )

tokenToInt :: Token -> Int
tokenToInt (Token tk) = fromIntegral tk

intToToken :: Int -> Token
intToToken = Token . fromIntegral

deriving newtype instance NFData Token
deriving newtype instance Storable Token

data HFTokenize = HFTokenize
  { hfModelID :: FilePath -- hf model ID or local path`
  , localFilesOnly :: Bool
  , trustRemoteCode :: Bool
  , textToTokenize :: Text
  , addSpecialTokens :: Bool
  }
  deriving ( Eq, Show, Ord, Read, Generic, Data, Typeable )

deriving anyclass instance NFData HFTokenize
