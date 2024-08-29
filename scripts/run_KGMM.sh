CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py\
    pde_instance.domain_dim=4\
    pde_instance=kinetic_fokker_planck\
    pde_instance.potential=GMM\
    pde_instance.n_steps=200\
    solver.train.batch_size_0T=2500\
    neural_network.hidden_dim=32\
    neural_network.layers=2\
    train.optimizer.learning_rate.initial=1e-2\
    pde_instance.total_evolving_time=2\
    train.optimizer.learning_rate.scheduling=cosine\
    backend.use_pmap_train=True