{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveGeneric #-}

module Pomnetic.Error
  ( PomneticError(..) )
  where

import Control.Exception
import Data.Data
import Data.Text ( Text )
import GHC.Generics
import Pomnetic.Types

data PomneticError
  = PomneticError String
  | TooLongText
  | AllTokensRejected
  | InvalidRegex !Text   -- text will contain human-readable message
  | InvalidToken !Token
  deriving ( Eq, Ord, Show, Data, Typeable, Generic )

instance Exception PomneticError
