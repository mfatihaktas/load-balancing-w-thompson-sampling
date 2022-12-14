import numpy
import simpy

from src.agent import (
    optimal as optimal_module,
    ts as ts_module,
)
from src.prob import random_variable
from src.sim import sim as sim_module
from src.sys import server as server_module

from src.utils.debug import *
from src.utils.plot import *


def test_AssignToLeastWorkLeft_vs_AssignWithThompsonSampling(
    env: simpy.Environment,
    num_servers: int,
    task_service_time_rv: random_variable.RandomVariable,
):
    num_tasks_to_recv = 200
    win_len = 100
    def sim_(arrival_rate: float):
        def assign_w_ts_sliding_win(server_list: list[server_module.Server]):
            return ts_module.AssignWithThompsonSampling_slidingWin(
                node_id_list=[s._id for s in server_list],
                win_len=win_len,
            )

        def assign_w_ts_sliding_win_for_each_node(server_list: list[server_module.Server]):
            return ts_module.AssignWithThompsonSampling_slidingWinForEachNode(
                node_id_list=[s._id for s in server_list],
                win_len=win_len,
            )

        def assign_to_least_loaded(server_list: list[server_module.Server]):
            return optimal_module.AssignToLeastWorkLeft(node_list=server_list)

        inter_task_gen_time_rv = random_variable.Exponential(mu=arrival_rate)

        sim_result_for_ts_sliding_win = sim_module.sim(
            env=env,
            num_servers=num_servers,
            inter_task_gen_time_rv=inter_task_gen_time_rv,
            task_service_time_rv=task_service_time_rv,
            num_tasks_to_recv=num_tasks_to_recv,
            sching_agent_given_server_list=assign_w_ts_sliding_win,
        )

        sim_result_for_ts_sliding_win_for_each_node = sim_module.sim(
            env=env,
            num_servers=num_servers,
            inter_task_gen_time_rv=inter_task_gen_time_rv,
            task_service_time_rv=task_service_time_rv,
            num_tasks_to_recv=num_tasks_to_recv,
            sching_agent_given_server_list=assign_w_ts_sliding_win_for_each_node,
        )

        sim_result_for_assign_to_least_loaded = sim_module.sim(
            env=env,
            num_servers=num_servers,
            inter_task_gen_time_rv=inter_task_gen_time_rv,
            task_service_time_rv=task_service_time_rv,
            num_tasks_to_recv=num_tasks_to_recv,
            sching_agent_given_server_list=assign_to_least_loaded,
        )

        return (
            sim_result_for_ts_sliding_win,
            sim_result_for_ts_sliding_win_for_each_node,
            sim_result_for_assign_to_least_loaded
        )

    # Run the sim
    arrival_rate_list = []
    ET_ts_sliding_win_list, ET_ts_sliding_win_for_each_node_list, ET_to_least_loaded_list = [], [], []
    std_T_ts_sliding_win_list, std_T_ts_sliding_win_for_each_node_list, std_T_to_least_loaded_list = [], [], []
    for arrival_rate in numpy.linspace(0.1, num_servers, num=4, endpoint=False):
        log(INFO, f">> arrival_rate= {arrival_rate}")
        arrival_rate_list.append(arrival_rate)

        (
            sim_result_for_ts_sliding_win,
            sim_result_for_ts_sliding_win_for_each_node,
            sim_result_for_assign_to_least_loaded
        ) = sim_(arrival_rate=arrival_rate)
        log(INFO, "",
            sim_result_for_ts_sliding_win=sim_result_for_ts_sliding_win,
            sim_result_for_ts_sliding_win_for_each_node=sim_result_for_ts_sliding_win_for_each_node,
            sim_result_for_assign_to_least_loaded=sim_result_for_assign_to_least_loaded,
        )

        ET_ts_sliding_win_list.append(sim_result_for_ts_sliding_win.ET)
        std_T_ts_sliding_win_list.append(sim_result_for_ts_sliding_win.std_T)

        ET_ts_sliding_win_for_each_node_list.append(sim_result_for_ts_sliding_win_for_each_node.ET)
        std_T_ts_sliding_win_for_each_node_list.append(sim_result_for_ts_sliding_win_for_each_node.std_T)

        ET_to_least_loaded_list.append(sim_result_for_assign_to_least_loaded.ET)
        std_T_to_least_loaded_list.append(sim_result_for_assign_to_least_loaded.std_T)

    plot.errorbar(arrival_rate_list, ET_ts_sliding_win_list, yerr=std_T_ts_sliding_win_list, color=next(dark_color_cycle), label="TS-SlidingWin", marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
    plot.errorbar(arrival_rate_list, ET_ts_sliding_win_for_each_node_list, yerr=std_T_ts_sliding_win_for_each_node_list, color=next(dark_color_cycle), label="TS-SlidingWinForEachNode", marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
    plot.errorbar(arrival_rate_list, ET_to_least_loaded_list, yerr=std_T_to_least_loaded_list, color=next(dark_color_cycle), label="AssignToLeastWorkLeft", marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    # Save the plot
    fontsize = 14
    plot.legend(fontsize=fontsize)
    plot.ylabel(r"$E[T]$", fontsize=fontsize)
    plot.xlabel(r"$\lambda$", fontsize=fontsize)
    plot.title(
        f"$N= {num_servers}$, "
        r"$X \sim \mathbf{Exp}(\lambda)$, "
        f"$S \sim {task_service_time_rv.to_latex()}$"
    )
    plot.gcf().set_size_inches(6, 4)
    plot.savefig("plot_AssignToLeastWorkLeft_vs_AssignWithThompsonSampling_ET_vs_lambda.png", bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")
