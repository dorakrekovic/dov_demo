import flwr as fl
import numpy as np
from flwr.common import parameters_to_ndarrays
# 🔄 Federated Server Strategy
class FaceRecognitionStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        print(f"🔄 Aggregating round {rnd} results...")
        if not results:
            return None, {}

        # Extract parameters from each client
       # print(parameters_to_ndarrays(results[0][1].parameters))
        embeddings = [np.array(parameters_to_ndarrays(res.parameters)) for _ , res in results]  # ✅ Correct unpacking

        # Compute Global Face Signature (Average of All Clients)
        global_signature = np.mean(embeddings, axis=0)
        print(f"✅ New Global Face Signature Computed!")

        return [global_signature], {}

# 🔄 Start Federated Server
fl.server.start_server(
    server_address="10.19.4.71:8080", 
    strategy=FaceRecognitionStrategy(), 
    config=fl.server.ServerConfig(num_rounds=5)
)

