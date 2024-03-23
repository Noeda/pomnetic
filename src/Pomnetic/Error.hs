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
  | NoRequestedTokenExists -- thrown when bos or eos was asked and model doesn ot have any.
  | TooLongText
  | AllTokensRejected
  | InvalidRegex !Text   -- text will contain human-readable message
  | InvalidToken !Token
  deriving ( Eq, Ord, Show, Data, Typeable, Generic )

instance Exception PomneticError
