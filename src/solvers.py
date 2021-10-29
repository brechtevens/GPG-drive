from src.settings import SolverSettings
import casadi as cs
import opengen as og
import sys
import os.path
import numpy as np
import panocpy as pa
from datetime import timedelta
import time


def get_ipopt_solver(problem, solver_settings:SolverSettings, bounds):
    problem.pop("g1")
    problem.pop("g2")
    solver = cs.nlpsol('Solver', 'ipopt', problem, {'verbose_init': False, 'print_time': False, 'ipopt':
                                                {'print_level': 0, 'tol': solver_settings.ipopt_tolerance,
                                                'acceptable_tol': solver_settings.ipopt_acceptable_tolerance,
                                                'max_cpu_time' : solver_settings.max_time}})
    # 'mu_strategy': 'adaptive', 'nlp_scaling_method': 'gradient-based', 'max_soc': 4}})

    def get_solver():
        def solver_function(x_old, params):
            cpu_begin = time.perf_counter()
            sol = solver(x0=x_old, p=params, lbx=bounds["lbx"], ubx=bounds["ubx"], lbg=bounds["lbg"], ubg=bounds["ubg"])
            time_elapsed = time.perf_counter() - cpu_begin
            return sol, time_elapsed
        return solver_function

    return get_solver()


def get_open_optimizer_name(solver_settings:SolverSettings, name, id=None, id2=None):
    optimizer_name = "py_" + name if solver_settings.open_use_python_bindings else "tcp_" + name
    if id is not None:
        optimizer_name += "_" + str(id)
        if id2 is not None:
            optimizer_name += "_" + str(id2)
    return optimizer_name


def get_panoc_optimizers_dir(solver_settings):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'build', solver_settings.open_directory_name)


def build_OpEn_problem(OpEn_problem, solver_settings:SolverSettings, optimizers_dir, id, id2, name):
    optimizer_name = get_open_optimizer_name(solver_settings, name, id, id2)
    solver_config = og.config.SolverConfiguration() \
                    .with_tolerance(solver_settings.panoc_initial_tolerance) \
                    .with_tolerance(solver_settings.panoc_tolerance) \
                    .with_delta_tolerance(solver_settings.panoc_delta_tolerance) \
                    .with_max_inner_iterations(solver_settings.panoc_max_inner_iterations) \
                    .with_max_outer_iterations(solver_settings.panoc_max_outer_iterations) \
                    .with_initial_penalty(solver_settings.panoc_initial_penalty) \
                    .with_penalty_weight_update_factor(solver_settings.panoc_penalty_weight_update_factor)\
                    .with_lbfgs_memory(solver_settings.panoc_lbfgsmem)\
                    .with_max_duration_micros(solver_settings.max_time * 10**6)

    if solver_settings.open_use_python_bindings:
        build_config = og.config.BuildConfiguration() \
            .with_build_python_bindings()
        meta = og.config.OptimizerMeta() \
            .with_optimizer_name(optimizer_name)
    else:
        tcp_config = og.config.TcpServerConfiguration('127.0.0.1', 8000 + 100*id2 + id)
        build_config = og.config.BuildConfiguration() \
            .with_tcp_interface_config(tcp_config)
        meta = og.config.OptimizerMeta() \
            .with_optimizer_name(optimizer_name)

    build_config.with_build_mode(solver_settings.panoc_build_mode)\
        .with_build_directory(optimizers_dir)
    builder = og.builder.OpEnOptimizerBuilder(OpEn_problem, meta, build_config, solver_config)
    builder.build()


def start_OpEn_solver(solver_settings:SolverSettings, optimizers_dir, id, id2, name):
    optimizer_name = get_open_optimizer_name(solver_settings, name, id, id2)
    if solver_settings.open_use_python_bindings:   
        sys.path.insert(1, os.path.join(optimizers_dir, optimizer_name))
        pysolver = __import__(optimizer_name)
        solver = pysolver.solver()

        return lambda x_old, params : solver.run(params, initial_guess = x_old)

    else:
        print("start controller TCP servers")
        # Start TCP server for calling optimizer
        solver = og.tcp.OptimizerTcpManager(os.path.join(optimizers_dir, optimizer_name))
        solver.start()
        solver.ping()
        print("TCP servers initialized")
        
        return lambda x_old, params : solver.call(params, initial_guess = x_old).get()

def get_OpEn_solver(problem, solver_settings:SolverSettings, bounds, id, id2=None, name="sol"):
    optimizers_dir = get_panoc_optimizers_dir(solver_settings)
    if solver_settings.panoc_rebuild_solver:

        # Initialize solver for 'OpEn'
        # Concatenate all variables for ALM solver
        bounds_x = og.constraints.Rectangle(bounds["lbx"], bounds["ubx"])
        bounds_g = og.constraints.Rectangle(bounds["lbg"], bounds["ubg"])
        bounds_g1 = og.constraints.Rectangle(bounds["lbg1"], bounds["ubg1"])
        bounds_g2 = og.constraints.Rectangle(bounds["lbg2"], bounds["ubg2"])

        # Create problem for current player and build the optimizer
        OpEn_problem = og.builder.Problem(problem["x"], problem["p"], problem["f"]) \
                       .with_constraints(bounds_x)
        if problem["g"].shape != (0, 0):          
            if solver_settings.panoc_use_alm:
                OpEn_problem.with_aug_lagrangian_constraints(problem["g"], bounds_g)
            else:
                OpEn_problem.with_penalty_constraints(problem["g2"])\
                    .with_aug_lagrangian_constraints(problem["g1"], bounds_g1) # TODO: include bounds_g2?

            # .with_penalty_constraints(cs.vertcat(*g, *player_constraints_g,
            #                                     *[cs.fmin(0.0, constraint) for constraint in player_constraints_h]))\
            # .with_aug_lagrangian_constraints(cs.vertcat(*h), og.constraints.Rectangle([0]*len(h), [float('inf')]*len(h)))

        build_OpEn_problem(OpEn_problem, solver_settings, optimizers_dir, id, id2, name)

    return start_OpEn_solver(solver_settings, optimizers_dir, id, id2, name)


def get_panocpy_solver(problem, solver_settings:SolverSettings, bounds, id, id2=None, name="panocpy"):
    # %% Build the problem for PANOC+ALM
    from tempfile import TemporaryDirectory

    name += '_' + str(id) 
    if id2 is not None:
        name += '_' + str(id2) 

    f_prob = cs.Function('f', [problem['x'], problem['p']], [problem['f']])
    g_prob = cs.Function('g', [problem['x'], problem['p']], [problem['g']])
    g1_prob = cs.Function('g1', [problem['x'], problem['p']], [problem['g1']])
    g2_prob = cs.Function('g2', [problem['x'], problem['p']], [problem['g2']])

    verbose = False

    panocparams = {
        "max_iter": solver_settings.panoc_max_inner_iterations,
        "max_time": timedelta(seconds=solver_settings.max_time),
        "print_interval": 100 if verbose else 0,
        # "stop_crit": pa.PANOCStopCrit.ProjGradUnitNorm,
        # "stop_crit": pa.PANOCStopCrit.FPRNorm,
        # "stop_crit": pa.PANOCStopCrit.ProjGradNorm,
        # "stop_crit": pa.PANOCStopCrit.ApproxKKT,
        "update_lipschitz_in_linesearch": True,
    }

    almparams = pa.ALMParams(
        max_iter=solver_settings.panoc_max_outer_iterations,
        max_time=timedelta(seconds=solver_settings.max_time),
        print_interval=10 if verbose else 0,
        preconditioning=False,
        ε=solver_settings.panoc_tolerance,
        δ=solver_settings.panoc_delta_tolerance,
        Σ_0=solver_settings.panoc_initial_penalty,
        Σ_max=1e12,
        Δ=solver_settings.panoc_penalty_weight_update_factor,
        max_total_num_retries=0
    )

    if solver_settings.panoc_use_alm:
        panocpy_prob = pa.generate_and_compile_casadi_problem(f_prob, g_prob, name=name)

        panocpy_prob.C.lowerbound = bounds["lbx"]
        panocpy_prob.C.upperbound = bounds["ubx"]
        panocpy_prob.D.lowerbound = bounds["lbg"]
        panocpy_prob.D.upperbound = bounds["ubg"]

        innersolver = pa.PANOCSolver(
            pa.PANOCParams(**panocparams),
            pa.LBFGSParams(memory=solver_settings.panoc_lbfgsmem),
        )

        solver = pa.ALMSolver(almparams, innersolver)

        def get_solver():
            def solver_function(x_old, params):
                panocpy_prob.param = params
                return solver(problem=panocpy_prob, x=x_old)
            return solver_function

    else:
        panocpy_prob = pa.generate_and_compile_casadi_problem_full(f_prob, g1_prob, g2_prob, name=name)

        panocpy_prob.C.lowerbound = bounds["lbx"]
        panocpy_prob.C.upperbound = bounds["ubx"]
        panocpy_prob.D1.lowerbound = bounds["lbg1"]
        panocpy_prob.D1.upperbound = bounds["ubg1"]
        panocpy_prob.D2.lowerbound = bounds["lbg2"]
        panocpy_prob.D2.upperbound = bounds["ubg2"]

        innersolver = pa.PANOCSolverFull(
            pa.PANOCParams(**panocparams),
            pa.LBFGSParams(memory=solver_settings.panoc_lbfgsmem),
        )

        solver = pa.ALMSolverFull(almparams, innersolver)

        def get_solver():
            def solver_function(x_old, params):
                panocpy_prob.param = params
                x_opt, mu_opt, stats = solver(problem=panocpy_prob, x=x_old)
                return x_opt, np.hstack([stats['penalty₂'],mu_opt]), stats 
            return solver_function

    return get_solver()