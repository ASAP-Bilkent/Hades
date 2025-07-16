from openfhe import *
from src.core import utils

class CKKSContext:
    def __init__(self, n_clients):
        is_multiparty = n_clients > 1
        utils.vprint("is_multiparty: ", is_multiparty)
        self.cc, self.depth = self._setup_self(is_multiparty)
        if is_multiparty:
            parties = [Party()]*n_clients
            for i in range(n_clients):
                parties[i].id = i
                if i == 0:
                    parties[i].kpShard = self.cc.KeyGen()
                else:
                    parties[i].kpShard = self.cc.MultipartyKeyGen(parties[0].kpShard.publicKey)
            for i in range(n_clients):
                if not parties[i].kpShard.good():
                    print(f"Key generation failed for party {i}!\n")
                    return 1

            # Generate collective public key
            secretKeys = []
            for i in range(n_clients):
                secretKeys.append(parties[i].kpShard.secretKey)
            self.keys = self.cc.MultipartyKeyGen(secretKeys)
        else:
            self.keys = self.cc.KeyGen()
        self.cc.EvalMultKeyGen(self.keys.secretKey)
        self.cc.EvalRotateKeyGen(self.keys.secretKey, [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
        self.cc.EvalBootstrapKeyGen(self.keys.secretKey, self.cc.GetRingDimension() // 2)

    def _setup_self(self,is_multiparty):
        secret_key_dist = SecretKeyDist.UNIFORM_TERNARY

        parameters = CCParamsCKKSRNS()
        parameters.SetSecretKeyDist(secret_key_dist)
        parameters.SetSecurityLevel(SecurityLevel.HEStd_NotSet)
        parameters.SetRingDim(1 << 13)

        if get_native_int() == 128:
            rescale_tech = ScalingTechnique.FIXEDAUTO
            dcrt_bits = 78
            first_mod = 89
        else:
            rescale_tech = ScalingTechnique.FLEXIBLEAUTO # FIXEDMANUAL like 2-3% faster
            dcrt_bits = 59
            first_mod = 60

        parameters.SetScalingModSize(dcrt_bits)
        parameters.SetScalingTechnique(rescale_tech)
        parameters.SetFirstModSize(first_mod)

        """
        https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/advanced-ckks-bootstrapping.cpp#L116
        /*  A4) Bootstrapping parameters.
        * We set a budget for the number of levels we can consume in bootstrapping for encoding and decoding, respectively.
        * Using larger numbers of levels reduces the complexity and number of rotation keys,
        * but increases the depth required for bootstrapping.
        * We must choose values smaller than ceil(log2(slots)). A level budget of {4, 4} is good for higher ring
        * dimensions (65536 and higher).
        */
        """
        level_budget = [2, 2]

        """
        /*  A5) Multiplicative depth.
        * The goal of bootstrapping is to increase the number of available levels we have, or in other words,
        * to dynamically increase the multiplicative depth. However, the bootstrapping procedure itself
        * needs to consume a few levels to run. We compute the number of bootstrapping levels required
        * using GetBootstrapDepth, and add it to levelsAvailableAfterBootstrap to set our initial multiplicative
        * depth.
        */
        """
        levels_available_after_bootstrap = 6
        depth = levels_available_after_bootstrap + FHECKKSRNS.GetBootstrapDepth(level_budget, secret_key_dist)
        #print(FHECKKSRNS.GetBootstrapDepth(level_budget, secret_key_dist))

        parameters.SetMultiplicativeDepth(depth)

        cc = GenCryptoContext(parameters)
        cc.Enable(PKESchemeFeature.PKE)
        cc.Enable(PKESchemeFeature.KEYSWITCH)
        cc.Enable(PKESchemeFeature.LEVELEDSHE)
        cc.Enable(PKESchemeFeature.ADVANCEDSHE)
        cc.Enable(PKESchemeFeature.FHE)
        if is_multiparty:
            cc.Enable(PKESchemeFeature.MULTIPARTY)

        cc.EvalBootstrapSetup(level_budget)     
                
        return cc, depth
    
class Party:
    def __init__(self, id, sharesPair, kpShard):
        self.id = id
        self.sharesPair = sharesPair
        self.kpShard = kpShard
    def __init__(self):
        self.id = None
        self.sharesPair = None
        self.kpShard = None
    def __str__(self):
        return f"Party {self.id}"