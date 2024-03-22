module Pomnetic.Safe
  ( safeFromIntegral )
  where

-- | Same as `fromIntegral` but will throw an error if the value is too large
-- to convert.
--
-- Specifically, it will test if conversion back to the original type is lossy.
safeFromIntegral :: (Integral a, Integral b) => a -> b
safeFromIntegral x =
  let y = fromIntegral x in
   if fromIntegral y == x
     then y
     else error "safeFromIntegral: lossy conversion"
