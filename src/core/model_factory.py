from src.core.models import Network, HENetwork, NoEncFusionNetwork, FusionNetwork

class ModelFactory:
    def __init__(self, args):
        self.args = args
    
    def get_single_network(self):
        if not self.args.enc:
            net = Network(
                dims=self.args.dims,
                args=self.args,
                network_label="PT"
            )
        else:
            net = HENetwork(
                dims=self.args.dims,
                args=self.args,
                network_label="CT"
            )
        
        return net
    
    def get_fusion_network(self):
        if not self.args.enc:
            net = NoEncFusionNetwork(
                args=self.args,
                dims_one=self.args.dims[0],
                dims_two=self.args.dims[1]
            )
        else:
            net = FusionNetwork(
                args=self.args,
                dims_one=self.args.dims[0],
                dims_two=self.args.dims[1]
            )
        
        return net 