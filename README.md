# PAKDD'2025

## Extensions
Current Unlearn process is to randomly retrain partial data in each client to complete unlearn model.

| Drawbacks:
- Low efficiency
- Specific client model performance degradation but couldn't point out which part of data be removed.
- Server lacks of ability to remove any one of client data.

| We could do:
- Support removing specific client for server / aggregation model.
- Add experiments to test global model that whether it's really be removed partial specific data.
- Backdoor Attack experiments
- Additional dataset experiments.


Related work

| Paper | Year | Conference | Link |
| ------| -----| ---------- | ---- |
| Fast-FedUL: A Training-Free Federated Unlearning with Provable Skew Resilience | 2024 | PKDD |  https://arxiv.org/pdf/2405.18040    |
|  Fast federated machine unlearning with nonlinear functional theory | 2023 | ICML |     |
