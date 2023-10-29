{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveGeneric #-}

module Pomnetic.Error
  ( PomneticError(..) )
  where

import Control.Exception
import Data.Data
import Data.Text ( Text )
import GHC.Generics

data PomneticError
  = PomneticError String
  | TooLongText
  | AllTokensRejected
  | InvalidRegex !Text   -- text will contain human-readable message
  deriving ( Eq, Ord, Show, Data, Typeable, Generic )

instance Exception PomneticError
