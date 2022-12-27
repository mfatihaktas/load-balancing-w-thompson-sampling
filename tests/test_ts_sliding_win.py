import simpy

from src.agent import ts as ts_module
from src.prob import random_variable
from src.sim import sim as sim_module
from src.sys import server as server_module

from src.utils.debug import *
from src.utils.plot import *


def test_AssignWithThompsonSampling(
    env: simpy.Environment,
    num_servers: int,
    inter_task_gen_time_rv: random_variable.RandomVariable,
    task_service_time_rv: random_variable.RandomVariable,
):
    def assign_w_ts_sliding_win(server_list: list[server_module.Server]):
        return ts_module.AssignWithThompsonSampling_slidingWin(
            node_list=server_list,
            win_len=100,
        )

    def assign_w_ts_reset_win_on_rare_event(server_list: list[server_module.Server]):
        return ts_module.AssignWithThompsonSampling_resetWinOnRareEvent(
            node_list=server_list,
            win_len=100,
            threshold_prob_rare=0.9,
        )

    sim_result = sim_module.sim(
        env=env,
        num_servers=num_servers,
        inter_task_gen_time_rv=inter_task_gen_time_rv,
        task_service_time_rv=task_service_time_rv,
        num_tasks_to_recv=100,
        sching_agent_given_server_list=assign_w_ts_reset_win_on_rare_event,
    )
    log(INFO, "", sim_result=sim_result)


def test_AssignWithThompsonSampling_slidingWin_vs_slidingWinForEachNode(
    env: simpy.Environment,
    num_servers: int,
    inter_task_gen_time_rv: random_variable.RandomVariable,
    task_service_time_rv: random_variable.RandomVariable,
):
    num_tasks_to_recv = 100
    def sim_(win_len: int):
        def ts_sliding_win(server_list: list[server_module.Server]):
            return ts_module.AssignWithThompsonSampling_slidingWin(
                node_list=server_list,
                win_len=win_len,
            )

        def ts_sliding_win_for_each_node(server_list: list[server_module.Server]):
            return ts_module.AssignWithThompsonSampling_slidingWinForEachNode(
                node_list=server_list,
                win_len=win_len,
            )

        sim_result_for_slidingWin = sim_module.sim(
            env=env,
            num_servers=num_servers,
            inter_task_gen_time_rv=inter_task_gen_time_rv,
            task_service_time_rv=task_service_time_rv,
            num_tasks_to_recv=num_tasks_to_recv,
            sching_agent_given_server_list=ts_sliding_win,
        )

        sim_result_for_slidingWinForEachNode = sim_module.sim(
            env=env,
            num_servers=num_servers,
            inter_task_gen_time_rv=inter_task_gen_time_rv,
            task_service_time_rv=task_service_time_rv,
            num_tasks_to_recv=num_tasks_to_recv,
            sching_agent_given_server_list=ts_sliding_win_for_each_node,
        )

        return sim_result_for_slidingWin, sim_result_for_slidingWinForEachNode

    # Run the sim
    win_len_list = []
    ET_slidingWin_list, ET_slidingWinForEachNode_list = [], []
    std_T_slidingWin_list, std_T_slidingWinForEachNode_list = [], []
    for win_len in [10, 100, 1000]:
        log(INFO, f">> win_len= {win_len}")
        win_len_list.append(win_len)

        sim_result_for_slidingWin, sim_result_for_slidingWinForEachNode = sim_(win_len=win_len)
        log(INFO, "", sim_result_for_slidingWin=sim_result_for_slidingWin, sim_result_for_slidingWinForEachNode=sim_result_for_slidingWinForEachNode)

        ET_slidingWin_list.append(sim_result_for_slidingWin.ET)
        std_T_slidingWin_list.append(sim_result_for_slidingWin.std_T)

        ET_slidingWinForEachNode_list.append(sim_result_for_slidingWinForEachNode.ET)
        std_T_slidingWinForEachNode_list.append(sim_result_for_slidingWinForEachNode.std_T)

    plot.errorbar(win_len_list, ET_slidingWin_list, yerr=std_T_slidingWin_list, color=next(dark_color_cycle), label="TS-SlidingWin", marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
    plot.errorbar(win_len_list, ET_slidingWinForEachNode_list, yerr=std_T_slidingWinForEachNode_list, color=next(dark_color_cycle), label="TS-SlidingWinForEachNode", marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    # Save the plot
    fontsize = 14
    plot.legend(fontsize=fontsize)
    plot.ylabel(r"$E[T]$", fontsize=fontsize)
    plot.xlabel(r"$w$", fontsize=fontsize)
    plot.title(
        f"$N= {num_servers}$, "
        f"$X \sim {inter_task_gen_time_rv.to_latex()}$, "
        f"$S \sim {task_service_time_rv.to_latex()}$"
    )
    plot.gcf().set_size_inches(6, 4)
    plot.savefig("plot_AssignWithThompsonSampling_slidingWin_vs_slidingWinForEachNode_ET_vs_w.png", bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")
