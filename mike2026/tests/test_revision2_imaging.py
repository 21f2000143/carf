import unittest
import numpy as np
from src.revision2_imaging import image_features

class ImageFeatureTests(unittest.TestCase):
    def setUp(self):
        self.pre=np.ones((5,6,7))*100
        self.mask=np.zeros_like(self.pre,dtype=bool); self.mask[1:4,1:5,1:6]=True

    def test_physical_volume_and_kinetics(self):
        f=image_features(self.pre,self.pre*2,self.pre*1.5,self.mask,[2,1,1])
        self.assertAlmostEqual(f['log_volume_mm3'],np.log1p(60*2))
        self.assertEqual(f['enhancement_mean'],1.)
        self.assertEqual(f['late_change_mean'],-.5)
        self.assertEqual(f['washout_fraction'],1.)
        self.assertEqual(f['texture_contrast'],0.)

    def test_scanner_signal_scale_invariance(self):
        a=image_features(self.pre,self.pre*2,self.pre*1.5,self.mask,[2,1,1])
        b=image_features(self.pre*7,self.pre*14,self.pre*10.5,self.mask,[2,1,1])
        np.testing.assert_allclose(list(a.values()),list(b.values()),atol=1e-12)

    def test_bad_alignment_rejected(self):
        with self.assertRaisesRegex(ValueError,'dimensions'):
            image_features(self.pre,self.pre[:-1],self.pre,self.mask,[1,1,1])

    def test_invalid_mask_rejected(self):
        with self.assertRaisesRegex(ValueError,'ten'):
            image_features(self.pre,self.pre,self.pre,np.zeros_like(self.mask),[1,1,1])

if __name__=='__main__': unittest.main()
