CUDA_VISIBLE_DEVICES=3
python main.py\
    pde_instance.domain_dim=2\
    pde_instance=kinetic_fokker_planck\
    pde_instance.potential=GMM\
    pde_instance.sample_mode=offline\
    neural_network.hidden_dim=32\
    neural_network.layers=2\
    train.optimizer.learning_rate.initial=1e-2\
    pde_instance.total_evolving_time=10\
    train.optimizer.learning_rate.scheduling=cosine\
    backend.use_pmap_train=False\
    seed=2\
    estimation_mode=parametric