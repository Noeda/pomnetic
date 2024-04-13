{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE TypeFamilies #-}

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
import qualified Data.Vector.Generic as VG
import qualified Data.Vector.Generic.Mutable as VGM
import qualified Data.Vector.Unboxed as VU
import Foreign.Storable
import GHC.Generics

-- Match with llama.cpp, where it is int32_t
newtype Token = Token Int32
  deriving ( Eq, Ord, Show, Data, Typeable, Generic )

-- boilerplate to let unboxed vectors use Token
newtype instance VU.MVector s Token = MV_Token (VU.MVector s Int32)
newtype instance VU.Vector    Token = V_Token  (VU.Vector    Int32)

instance VGM.MVector VU.MVector Token where
  {-# INLINE basicLength #-}
  basicLength (MV_Token v) = VGM.basicLength v

  {-# INLINE basicUnsafeSlice #-}
  basicUnsafeSlice i n (MV_Token v) = MV_Token $ VGM.basicUnsafeSlice i n v

  {-# INLINE basicOverlaps #-}
  basicOverlaps (MV_Token v1) (MV_Token v2) = VGM.basicOverlaps v1 v2

  {-# INLINE basicUnsafeNew #-}
  basicUnsafeNew n = MV_Token <$> VGM.basicUnsafeNew n

  {-# INLINE basicInitialize #-}
  basicInitialize (MV_Token v) = VGM.basicInitialize v

  {-# INLINE basicUnsafeRead #-}
  basicUnsafeRead (MV_Token v) i = Token <$> VGM.basicUnsafeRead v i

  {-# INLINE basicUnsafeWrite #-}
  basicUnsafeWrite (MV_Token v) i (Token tk) = VGM.basicUnsafeWrite v i tk

instance VG.Vector VU.Vector Token where
  {-# INLINE basicLength #-}
  basicLength (V_Token v) = VG.basicLength v

  {-# INLINE basicUnsafeSlice #-}
  basicUnsafeSlice i n (V_Token v) = V_Token $ VG.basicUnsafeSlice i n v

  {-# INLINE basicUnsafeFreeze #-}
  basicUnsafeFreeze (MV_Token v) = V_Token <$> VG.basicUnsafeFreeze v

  {-# INLINE basicUnsafeThaw #-}
  basicUnsafeThaw (V_Token v) = MV_Token <$> VG.basicUnsafeThaw v

  {-# INLINE basicUnsafeIndexM #-}
  basicUnsafeIndexM (V_Token v) i = Token <$> VG.basicUnsafeIndexM v i

  {-# INLINE basicUnsafeCopy #-}
  basicUnsafeCopy (MV_Token mv) (V_Token v) = VG.basicUnsafeCopy mv v

tokenToInt :: Token -> Int
tokenToInt (Token tk) = fromIntegral tk

intToToken :: Int -> Token
intToToken = Token . fromIntegral

deriving newtype instance VU.Unbox Token
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
