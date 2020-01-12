import sys
import math
import numpy as np


class Rand48(object):
    def __init__(self, seed):
        self.n = seed

    def srand(self, seed):
        self.n = (seed << 16) + 0x330e

    def next(self):
        self.n = (25214903917 * self.n + 11) & (2**48 - 1)
        return self.n

    def drand(self):
        return self.next() / 2**48


class Process:
    def __init__(self, at, nb):
        self.id = ""
        self.arrive_time = int(at)
        self.num_burst = int(nb)
        self.burst_list = list()
        self.tau_list = list()
        self.resume_time = -1
        self.complete_cpu_time = -1
        self.complete_io_time = -1
        self.cur_cpu_burst = 0
        self.cur_io_burst = 0
        self.waiting = False
        self.remaining_time = 0
        self.running_time = 0
        self.cur_burst = 0
        self.add_to_queue_time = -1

    def add_burst(self, CPU_b, IO_b):
        self.burst_list.append((CPU_b, IO_b))

    def add_tau(self, tau):
        self.tau_list.append(int(math.ceil(tau)))

    def set_resume(self, current_time, wait_time):
        self.resume_time = int(current_time + wait_time)

    def set_complete_cpu_time(self, current_time):
        self.complete_cpu_time = int(current_time + self.burst_list[self.cur_burst][0])

    def set_complete_io_time(self, current_time):
        self.complete_io_time = int(current_time + self.burst_list[self.cur_burst][1])

    def set_complete_cpu_time_SRT(self, current_time):
        self.complete_cpu_time = int(current_time + self.burst_list[self.cur_cpu_burst][0] - self.running_time)

    def set_complete_io_time_SRT(self, current_time):
        self.complete_io_time = int(current_time + self.burst_list[self.cur_io_burst][1])

    def add_cur_burst(self):
        self.cur_burst += 1


process_list = list()
rand_obj = Rand48(2)


def exp_rand(l, bound):
    global rand_obj

    while True:
        r = rand_obj.drand()
        x = - np.log(r) / l

        if x <= bound:
            return x
        # print("=================== Out of bound {:.6f} =====================".format(x))


def rand_config(lamb, exp_limit, num_process, alph):
    global process_list
    global rand_obj

    for p in range(num_process):
        arrival_time = exp_rand(lamb, exp_limit)
        arrival_time = math.floor(arrival_time)

        num_burst = rand_obj.drand()
        num_burst = math.floor(num_burst * 100) + 1

        new_p = Process(arrival_time, num_burst)

        new_p.add_tau(1/lamb)

        for i in range(num_burst):
            cpu_bt = exp_rand(lamb, exp_limit)
            cpu_bt = math.ceil(cpu_bt)

            if i != num_burst - 1:
                io_bt = exp_rand(lamb, exp_limit)
                io_bt = math.ceil(io_bt)
            else:
                io_bt = 0

            new_p.add_burst(cpu_bt, io_bt)

            if i != 0:
                cur_tau = alph * new_p.burst_list[i - 1][0] + (1 - alph) * new_p.tau_list[i - 1]
                new_p.add_tau(cur_tau)

        new_p.remaining_time = new_p.tau_list[0]
        process_list.append(new_p)


# Use this function to print overall info
def print_process_info(mode):
    global process_list

    p_index = 65

    for p in process_list:
        p.id = chr(p_index)
        if p.num_burst > 1:
            print("Process {} [NEW] (arrival time {} ms) {} CPU bursts".format(chr(p_index), p.arrive_time,
                                                                               p.num_burst), end="")
        else:
            print("Process {} [NEW] (arrival time {} ms) {} CPU burst".format(chr(p_index), p.arrive_time,
                                                                               p.num_burst), end="")
        if mode.upper() == "SRT" or mode.upper() == "SJF":
            print(" (tau {}ms)".format(p.tau_list[0]))
        else:
            print()
        p_index += 1


def print_queue(queue):
    print("[Q ", end="")

    if len(queue) == 0:
        print("<empty>]")
        return

    for q in range(len(queue)):
        print("{}".format(queue[q].id), end="")
        if q != len(queue) - 1:
            print(" ", end="")
    print("]")


def SRT_add_to_queue(queue, p):
    if len(queue) == 0:
        queue.append(p)
    else:
        for i in range(len(queue)):
            if p.remaining_time < queue[i].remaining_time:
                queue.insert(i, p)
                return
            elif p.remaining_time == queue[i].remaining_time:
                if p.id < queue[i].id:
                    queue.insert(i, p)
                    return
        queue.append(p)


def increase_wait_time(p, wait_list, n):
    global process_list
    for i in range(len(process_list)):
        if process_list[i].id == p.id:
            wait_list[i] += n
            break


def set_turnaround(p, turn_list, time, mode):
    global process_list
    for i in range(len(process_list)):
        if process_list[i].id == p.id:
            if mode:
                n = 0
                if turn_list[i][p.cur_cpu_burst][n] < 0:
                    turn_list[i][p.cur_cpu_burst][n] = time
                # sys.stderr.write("({},{}) ==> {} ==> {}\n".format(i, p.cur_cpu_burst, turn_list[i][p.cur_cpu_burst], n))
            else:
                n = 1
                turn_list[i][p.cur_cpu_burst - 1][n] = time
                # sys.stderr.write("({},{}) ==> {} ==> {}\n".format(i, p.cur_cpu_burst, turn_list[i][p.cur_cpu_burst], n))

            break


def print_time(mode, time_list):
    sys.stderr.write("Algorithm {}\n".format(mode))
    sys.stderr.write("-- average CPU burst time: {0:.3f} ms\n".format(time_list[0]))
    sys.stderr.write("-- average wait time: {0:.3f} ms\n".format(time_list[1]))
    sys.stderr.write("-- average turnaround time: {0:.3f} ms\n".format(time_list[2]))
    sys.stderr.write("-- total number of context switches: {}\n".format(time_list[3]))
    sys.stderr.write("-- total number of preemptions: {}\n".format(time_list[4]))


def FCFS(switch_time):
    global process_list
    queue = list()
    new_process_list = process_list[:]
    print_process_info("FCFS")
    empty_CPU = True
    click = True
    timer = 0
    print("time 0ms: Simulator started for FCFS [Q <empty>]")
    process_count = 0
    end_time = -1
    CPU_ava_time = -1
    switch_in_time = 0
    switching_queue = list()    

    cpu_burst_sum = 0
    cpu_burst_count = 0
    wait_time_sum = 0
    total_cont_switch = 0
    while True:
        if len(switching_queue) != 0 and empty_CPU and timer >= CPU_ava_time:
            if switching_queue[0].resume_time <= timer:
                cpu_burst_sum += switching_queue[0].burst_list[switching_queue[0].cur_burst][0]
                cpu_burst_count += 1
                total_cont_switch += 1
                print("time {}ms: Process {} started using the CPU for {}ms burst ".format(timer, switching_queue[0].id, switching_queue[0].burst_list[switching_queue[0].cur_burst][0]), end="")
                switching_queue[0].set_complete_cpu_time(timer)
                print_queue(queue)
                empty_CPU = False
                del switching_queue[0]

        for p in new_process_list:    

            if p.complete_cpu_time == timer and p.cur_burst < p.num_burst - 1:  # A burst is completed but the process is not termintated
                BURST = ("burst", "bursts")
                choice = -1
                if p.num_burst - p.cur_burst - 1 == 1:
                    choice = 0
                else:
                    choice = 1
                print("time {}ms: Process {} completed a CPU burst; {} {} to go ".format(timer, p.id, p.num_burst - p.cur_burst - 1, BURST[choice]), end="")
                print_queue(queue)
                CPU_ava_time = timer + switch_time
                p.set_complete_io_time(timer + switch_time / 2)
                switch_in_time = timer + switch_time / 2
                print("time {}ms: Process {} switching out of CPU; will block on I/O until time {}ms ".format(timer,
                                                                                                                 p.id,
                                                                                                                 p.complete_io_time),
                      end="")
                print_queue(queue)
                empty_CPU = True
                if len(queue) != 0:  # It takes time when new process switching into CPU
                    queue[0].set_resume(timer, switch_time)

            if p.complete_cpu_time == timer and p.cur_burst == p.num_burst - 1:  # Process terminates
                print("time {}ms: Process {} terminated ".format(timer, p.id), end="")
                print_queue(queue)
                CPU_ava_time = timer + switch_time
                switch_in_time = timer + switch_time / 2
                empty_CPU = True
                process_count += 1
                if len(queue) != 0:
                    queue[0].set_resume(timer, switch_time)

            
                
        for p in new_process_list:
                
            if p.arrive_time == timer:  # Process arrive
                queue.append(p)
                print("time {}ms: Process {} arrived; added to ready queue ".format(timer, p.id), end="")
                print_queue(queue)
                p.set_resume(timer, switch_time / 2)          
            
            if p.complete_io_time == timer:  # Process ends I/O
                queue.append(p)
                p.set_resume(timer, switch_time / 2)  # set a default switch time
                print("time {}ms: Process {} completed I/O; added to ready queue ".format(timer, p.id), end="")
                print_queue(queue)
                p.add_cur_burst()            
                
        if process_count == len(new_process_list) and click:
            end_time = timer + switch_time / 2
            click = False
            
        if switch_in_time == timer:
            switching_queue_open = True
    
        if switch_in_time <= timer and switching_queue_open and len(queue) != 0:
            switching_queue.append(queue[0])
            del queue[0]
            switching_queue_open = False        

        if end_time == timer:
            print("time {}ms: Simulator ended for FCFS [Q <empty>]".format(timer))
            break

        timer += 1
        wait_time_sum += len(queue)

    wait_time_sum -= total_cont_switch * switch_time / 2

    return cpu_burst_sum / cpu_burst_count, wait_time_sum / cpu_burst_count, (
                wait_time_sum + cpu_burst_sum) / cpu_burst_count + switch_time, total_cont_switch, 0


def cur_tau(a):
    return math.ceil(a.tau_list[a.cur_burst]), a.id


def SJF(switch_time):
    global process_list
    queue = list()
    new_process_list = process_list[:]

    print_process_info("SJF")
    empty_CPU = True
    click = True
    switching_queue_open = True
    timer = 0
    print("time 0ms: Simulator started for SJF [Q <empty>]")
    process_count = 0
    end_time = -1
    CPU_ava_time = -1
    switch_in_time = 0
    switching_queue = list()

    cpu_burst_sum = 0
    cpu_burst_count = 0
    wait_time_sum = 0
    total_cont_switch = 0

    for p in new_process_list:
        p.resume_time = -1
        p.complete_cpu_time = -1
        p.complete_io_time = -1
        p.cur_burst = 0

    while True:
        if len(switching_queue) != 0 and empty_CPU and timer >= CPU_ava_time:
            if switching_queue[0].resume_time <= timer:
                cpu_burst_sum += switching_queue[0].burst_list[switching_queue[0].cur_burst][0]
                cpu_burst_count += 1
                total_cont_switch += 1
                print("time {}ms: Process {} (tau {}ms) started using the CPU for {}ms burst ".format(timer, switching_queue[0].id, math.ceil(cur_tau(switching_queue[0])[0]), switching_queue[0].burst_list[switching_queue[0].cur_burst][0]), end="")
                switching_queue[0].set_complete_cpu_time(timer)
                print_queue(queue)
                empty_CPU = False
                del switching_queue[0]

        for p in new_process_list:
            if p.complete_cpu_time == timer and p.cur_burst < p.num_burst - 1:
                BURST=("burst","bursts")
                choice=-1
                if p.num_burst - p.cur_burst - 1==1:
                    choice=0
                else:
                    choice=1
                print("time {}ms: Process {} (tau {}ms) completed a CPU burst; {} {} to go ".format(timer, p.id, math.ceil(cur_tau(p)[0]), p.num_burst - p.cur_burst - 1, BURST[choice]), end="")
                print_queue(queue)
                print("time {}ms: Recalculated tau = {}ms for process {} ".format(timer, math.ceil(p.tau_list[p.cur_burst + 1]), p.id), end="")
                print_queue(queue)
                p.set_complete_io_time(timer + switch_time / 2)
                print("time {}ms: Process {} switching out of CPU; will block on I/O until time {}ms ".format(timer, p.id, int(p.complete_io_time)), end="")
                print_queue(queue)
                empty_CPU = True
                CPU_ava_time = timer + switch_time
                switch_in_time = timer + switch_time / 2
                if len(queue) != 0:
                    queue[0].set_resume(timer, switch_time)

            if p.complete_cpu_time == timer and p.cur_burst == p.num_burst - 1:
                print("time {}ms: Process {} terminated ".format(timer, p.id), end="")
                print_queue(queue)
                CPU_ava_time = timer + switch_time
                switch_in_time = timer + switch_time / 2
                empty_CPU = True
                process_count += 1
                if len(queue) != 0:
                    queue[0].set_resume(timer, switch_time)

            

        for p in new_process_list:
            if p.arrive_time == timer:
                queue.append(p)
                queue.sort(key=cur_tau)  # sort ready queue when new process added
                print("time {}ms: Process {} (tau {}ms) arrived; added to ready queue ".format(timer, p.id, math.ceil(cur_tau(p)[0])), end="")
                print_queue(queue)
                p.set_resume(timer, switch_time / 2)   
                
            if p.complete_io_time == timer:
                p.add_cur_burst()
                queue.append(p)
                p.set_resume(timer, switch_time / 2)  # set a default switch time
                print("time {}ms: Process {} (tau {}ms) completed I/O; added to ready queue ".format(timer, p.id, math.ceil(cur_tau(p)[0])), end="")
                queue.sort(key=cur_tau)  # sort the ready queue when new process added
                print_queue(queue)            
                
        if process_count == len(new_process_list) and click:
            end_time = timer + switch_time / 2
            click = False

        if switch_in_time == timer:
            switching_queue_open = True

        if switch_in_time <= timer and switching_queue_open and len(queue) != 0:
            switching_queue.append(queue[0])
            del queue[0]
            switching_queue_open = False

        if end_time == timer:
            print("time {}ms: Simulator ended for SJF [Q <empty>]".format(timer))
            break

        timer += 1
        wait_time_sum += len(queue)

    return cpu_burst_sum / cpu_burst_count, wait_time_sum / cpu_burst_count, (
                wait_time_sum + cpu_burst_sum) / cpu_burst_count + switch_time, total_cont_switch, 0


def SRT(num_process, switch_time):
    global process_list
    queue = list()
    print_process_info("SRT")
    new_process_list = process_list[:]

    for p in new_process_list:
        p.resume_time = -1
        p.complete_cpu_time = -1
        p.complete_io_time = -1
        p.cur_burst = 0

    timer = 0
    print("time 0ms: Simulator started for SRT [Q <empty>]")

    stop_count = 0
    terminate_count = -1

    preem_count = 0
    switch_count = 0

    total_cpu_burst = 0
    total_io_burst = 0
    num_burst = 0
    for p in process_list:
        for burst in p.burst_list:
            total_cpu_burst += burst[0]
            total_io_burst += burst[1]
            num_burst += 1
    avg_cpu_burst = total_cpu_burst/num_burst

    turn_around = list()
    wait_time = 0
    total_num_burst = 0

    for i in range(num_process):
        num_burst = new_process_list[i].num_burst
        total_num_burst += num_burst
        p_list = list()

        for j in range(num_burst):
            p_list.append([-1, -1])
        turn_around.append(p_list)

    cur_running = None

    clear_time = switch_time/2

    while True:

        if stop_count >= num_process:
            terminate_count = int(timer + switch_time/2 - 1)
            stop_count = -1

        if terminate_count != -1 and timer >= terminate_count:
            print("time {}ms: Simulator ended for SRT ".format(timer), end='')
            print_queue(queue)
            break

        if cur_running is not None:

            if cur_running.resume_time - timer == switch_time/2 - 1:
                queue.pop(queue.index(cur_running))
            elif cur_running.resume_time - timer > switch_time/2 - 1:
                wait_time += 1

            if cur_running.complete_cpu_time == timer:
                cur_running.running_time = 0
                cur_running.cur_cpu_burst += 1
                if cur_running.num_burst - cur_running.cur_cpu_burst > 1:
                    print("time {}ms: Process {} (tau {}ms) completed a CPU burst; {} bursts to go "
                          .format(timer, cur_running.id, cur_running.tau_list[cur_running.cur_cpu_burst - 1],
                                  len(cur_running.burst_list) - cur_running.cur_cpu_burst),
                          end="")
                    set_turnaround(cur_running, turn_around, timer + int(switch_time / 2), False)
                elif cur_running.num_burst - cur_running.cur_cpu_burst == 1:
                    print("time {}ms: Process {} (tau {}ms) completed a CPU burst; {} burst to go "
                          .format(timer, cur_running.id, cur_running.tau_list[cur_running.cur_cpu_burst - 1],
                                  len(cur_running.burst_list) - cur_running.cur_cpu_burst),
                          end="")
                    set_turnaround(cur_running, turn_around, timer + int(switch_time/2), False)
                if cur_running.num_burst - cur_running.cur_cpu_burst != 0:
                    print_queue(queue)

                if cur_running.cur_cpu_burst != cur_running.num_burst:
                    print("time {}ms: Recalculated tau = {}ms for process {} "
                          .format(timer, cur_running.tau_list[cur_running.cur_cpu_burst], cur_running.id), end='')
                    cur_running.remaining_time = cur_running.tau_list[cur_running.cur_cpu_burst]
                    print_queue(queue)

                    cur_running.set_complete_io_time_SRT(timer + switch_time/2)
                    print("time {}ms: Process {} switching out of CPU; will block on I/O until time {}ms "
                          .format(timer, cur_running.id, cur_running.complete_io_time), end="")
                    print_queue(queue)
                    clear_time = timer + switch_time

                    cur_running = None
                    # switch_count += 1
                else:
                    print("time {}ms: Process {} terminated ".format(timer, cur_running.id), end="")
                    print_queue(queue)
                    # cur_running.terminate_time = timer
                    clear_time = timer + switch_time
                    stop_count += 1
                    set_turnaround(cur_running, turn_around, timer + int(switch_time/2), False)
                    cur_running = None

            else:
                if cur_running.resume_time < timer:
                    # sys.stderr.write("{} {}\n".format(cur_running.running_time, timer))
                    cur_running.running_time += 1
                    cur_running.remaining_time -= 1

        # sys.stderr.write("Timer: {}\n".format(timer))
        for p in new_process_list:

            if p.add_to_queue_time == timer:
                SRT_add_to_queue(queue, p)

            if p.arrive_time == timer:
                SRT_add_to_queue(queue, p)

                if cur_running is None or p.remaining_time > cur_running.remaining_time or (p.remaining_time ==
                                                                                            cur_running.remaining_time
                                                                                            and p.id > cur_running.id):

                    print("time {}ms: Process {} (tau {}ms) arrived; added to ready queue "
                          .format(timer, p.id, p.tau_list[p.cur_cpu_burst]), end="")

                    print_queue(queue)
                    p.waiting = True
                    set_turnaround(p, turn_around, timer, True)

                elif p.id == queue[0].id:
                    if cur_running.resume_time <= timer:
                        print("time {}ms: Process {} (tau {}ms) arrived; preempting {} "
                              .format(timer, p.id, p.tau_list[p.cur_cpu_burst], cur_running.id), end="")
                        print_queue(queue)
                        cur_running.add_to_queue_time = timer + switch_time/2
                        set_turnaround(p, turn_around, timer, True)
                        p.waiting = True
                        preem_count += 1
                        p.set_resume(timer, switch_time)
                    else:
                        print("time {}ms: Process {} (tau {}ms) arrived; added to ready queue "
                              .format(timer, p.id, p.tau_list[p.cur_cpu_burst]), end="")
                        print_queue(queue)
                        p.waiting = True
                        set_turnaround(p, turn_around, timer, True)

        if cur_running is not None:
            if cur_running.resume_time == timer:

                print("time {}ms: Process {} (tau {}ms) started using the CPU with {}ms burst remaining "
                      .format(timer, cur_running.id, cur_running.tau_list[cur_running.cur_cpu_burst], cur_running.burst_list[cur_running.cur_cpu_burst][0] - cur_running.running_time),
                      end="")
                print_queue(queue)
                cur_running.set_complete_cpu_time_SRT(timer)

                switch_count += 1

                if cur_running.remaining_time <= 0:
                    cur_running.remaining_time = cur_running.tau_list[cur_running.cur_cpu_burst]

                if len(queue) != 0:
                    if queue[0].remaining_time < cur_running.remaining_time or (queue[0].remaining_time == cur_running.remaining_time and queue[0].id < cur_running.id):
                        print("time {}ms: Process {} (tau {}ms) will preempt {} ".format(timer, queue[0].id, queue[0].tau_list[queue[0].cur_cpu_burst], cur_running.id), end='')
                        print_queue(queue)
                        preem_count += 1
                        queue[0].set_resume(timer, switch_time)
                        cur_running.add_to_queue_time = timer + switch_time/2
                        temp = cur_running
                        cur_running = queue[0]
                        clear_time = timer + switch_time

        for p in new_process_list:
            if p.complete_io_time == timer:
                p.cur_io_burst += 1
                SRT_add_to_queue(queue, p)

                if cur_running is not None and (p.remaining_time < cur_running.remaining_time or (p.remaining_time == cur_running.remaining_time-1 and p.id < cur_running.id)):
                    if p.id == queue[0].id:
                        if cur_running.resume_time <= timer:
                            print("time {}ms: Process {} (tau {}ms) completed I/O; preempting {} "
                                  .format(timer, p.id, p.tau_list[p.cur_cpu_burst], cur_running.id), end='')
                            print_queue(queue)

                            set_turnaround(p, turn_around, timer, True)
                            preem_count += 1

                            p.set_resume(timer, switch_time)
                            cur_running.set_resume(timer, switch_time)
                            cur_running.complete_cpu_time = timer
                            cur_running.add_to_queue_time = timer + switch_time/2
                            clear_time = timer + switch_time
                            cur_running = None

                        else:
                            if cur_running.resume_time - timer >= switch_time/2:
                                # SRT_add_to_queue(queue, cur_running)
                                cur_running = None
                            print("time {}ms: Process {} (tau {}ms) completed I/O; added to ready queue "
                                  .format(timer, p.id, p.tau_list[p.cur_cpu_burst]), end="")
                            print_queue(queue)
                            set_turnaround(p, turn_around, timer, True)
                            p.waiting = True

                else:

                    print("time {}ms: Process {} (tau {}ms) completed I/O; added to ready queue "
                          .format(timer, p.id, p.tau_list[p.cur_cpu_burst]), end='')
                    print_queue(queue)
                    set_turnaround(p, turn_around, timer, True)
                    p.waiting = True

        if cur_running is None:
            if len(queue) != 0:
                queue[0].waiting = False
                cur_running = queue[0]

                if cur_running.resume_time < timer:
                    if timer + switch_time/2 <= clear_time:
                        cur_running.resume_time = clear_time
                    else:
                        cur_running.resume_time = timer + switch_time/2

        for wp in queue:
            if wp.id != cur_running.id:
                wait_time += 1
        timer += 1

    return [avg_cpu_burst, turn_around, wait_time, total_num_burst, preem_count, switch_count]


def RR(slice_t, switch, flag):
    global process_list
    new_list = process_list[:]
    time = 0
    in_t = 0
    out_t = 0
    remain_t = slice_t
    burst_p = ''
    ind = 0
    ready_q = []

    # calculate time....
    burst_t = 0
    wait_t = 0
    tt_t = 0
    num_switch = 0
    num_prem = 0
    for p in new_list:
        for b in p.burst_list:
            burst_t += b[0]

            # keep track of burst index, process status, block until time and preemption
    index = []
    for p in new_list:
        index.append(p.num_burst)
    stat = []
    for p in new_list:
        stat.append("NA")
    preempted = []
    for p in new_list:
        pre = False
        preempted.append(pre)
    until = []
    for p in new_list:
        until.append(0)

    # keep track of arrive time and end time for each burst
    end = []
    arrive = []
    for p in new_list:
        end.append([0] * p.num_burst)
        arrive.append([0] * p.num_burst)

    print_process_info("RR")
    print("time 0ms: Simulator started for RR [Q <empty>]")

    while True:
        for p in new_list:
            # when process status: running
            if p.id == burst_p and stat[ord(p.id) - 65] == "R":
                remain_t -= 1
                ind = p.num_burst - index[ord(p.id) - 65]
                m = p.burst_list[ind][0] - 1
                n = p.burst_list[ind][1]
                p.burst_list[ind] = (m, n)

                if p.burst_list[ind][0] == 0:
                    index[ord(p.id) - 65] -= 1
                    if index[ord(p.id) - 65] == 0:
                        print("time {}ms: Process {} terminated ".format(time, p.id), end="")
                        print_queue(ready_q)
                        stat[ord(p.id) - 65] = "CSend"
                        out_t = time

                        ind = p.num_burst - 1
                        end[ord(p.id) - 65][ind] = time
                    else:
                        out_t = time

                        ind = p.num_burst - index[ord(p.id) - 65]
                        end[ord(p.id) - 65][ind - 1] = time
                        if index[ord(p.id) - 65] == 1:
                            print('time {}ms: Process {} completed a CPU burst; {} burst to go '.format(time, p.id,
                                                                                                        index[ord(
                                                                                                            p.id) - 65]),
                                  end="")
                        else:
                            print('time {}ms: Process {} completed a CPU burst; {} bursts to go '.format(time, p.id,
                                                                                                         index[ord(
                                                                                                             p.id) - 65]),
                                  end="")
                        print_queue(ready_q)

                        until[ord(p.id) - 65] = int(p.burst_list[ind - 1][1] + (switch / 2) + time)
                        print('time {}ms: Process {} switching out of CPU; will block on I/O until time {}ms '.format(
                            time, p.id, until[ord(p.id) - 65]), end="")
                        print_queue(ready_q)

                        stat[ord(p.id) - 65] = "CSout"
                        preempted[ord(p.id) - 65] = False

        for p in new_list:
            # when time slice expired
            ind = p.num_burst - index[ord(p.id) - 65]
            if remain_t == 0 and p.id == burst_p and stat[ord(p.id) - 65] != "CSout":
                if len(ready_q) != 0:
                    preempted[ord(p.id) - 65] = True
                    num_prem += 1
                    stat[ord(p.id) - 65] = "CSout"
                    out_t = time

                    print('time {}ms: Time slice expired; process {} preempted with {}ms to go '.format(time, p.id,
                                                                                                        p.burst_list[
                                                                                                            ind][0]),
                          end="")
                    print_queue(ready_q)
                else:
                    print(
                        'time {}ms: Time slice expired; no preemption because ready queue is empty [Q <empty>]'.format(
                            time))
                remain_t = slice_t

        for p in new_list:
            # when process status: context switch out
            if stat[ord(p.id) - 65] == "CSout":
                if out_t + (switch / 2) == time:
                    burst_p = ""
                    if preempted[ord(p.id) - 65]:
                        stat[ord(p.id) - 65] = "A"
                        if (not RR_flag):
                            ready_q.append(p)
                        else:
                            ready_q.insert(1, p)
                    else:
                        stat[ord(p.id) - 65] = "B"
                    remain_t = slice_t

        for p in new_list:
            # when process status: context switch in
            if stat[ord(p.id) - 65] == "CSin":
                if (in_t + (switch / 2)) == time:
                    num_switch += 1
                    remain_t = slice_t

                    stat[ord(p.id) - 65] = "R"
                    ind = p.num_burst - index[ord(p.id) - 65]
                    if preempted[ord(p.id) - 65] == False:
                        print("time {}ms: Process {} started using the CPU for {}ms burst ".format(time, p.id,
                                                                                                   p.burst_list[ind][
                                                                                                       0]), end="")
                        print_queue(ready_q)
                    else:
                        print(
                            "time {}ms: Process {} started using the CPU with {}ms burst remaining ".format(time, p.id,
                                                                                                            p.burst_list[
                                                                                                                ind][
                                                                                                                0]),
                            end="")
                        print_queue(ready_q)

        for p in new_list:
            # when process status: block
            if stat[ord(p.id) - 65] == "B":
                if until[ord(p.id) - 65] == time:
                    if (not RR_flag):
                        ready_q.append(p)
                    else:
                        ready_q.insert(0, p)
                    print("time {}ms: Process {} completed I/O; added to ready queue ".format(time, p.id), end="")
                    print_queue(ready_q)
                    stat[ord(p.id) - 65] = "A"

                    ind = p.num_burst - index[ord(p.id) - 65]
                    arrive[ord(p.id) - 65][ind] = time

        for p in new_list:
            # when process status: not arrive
            if stat[ord(p.id) - 65] == "NA":
                if p.arrive_time == time:
                    stat[ord(p.id) - 65] = "A"

                    ind = p.num_burst - index[ord(p.id) - 65]
                    arrive[ord(p.id) - 65][ind] = time
                    if (not RR_flag):
                        ready_q.append(p)
                    else:
                        ready_q.insert(0, p)
                    print("time {}ms: Process {} arrived; added to ready queue ".format(time, p.id), end="")
                    print_queue(ready_q)

        for p in new_list:
            # when process status: context switch terminate
            if stat[ord(p.id) - 65] == "CSend":
                if out_t + (switch / 2) == time:
                    burst_p = ''
                    stat[ord(p.id) - 65] = "Term"

        for p in new_list:
            # when process status: ready
            if stat[ord(p.id) - 65] == "A":
                if p.id == ready_q[0].id and burst_p == '':
                    in_t = time
                    stat[ord(p.id) - 65] = "CSin"
                    burst_p = p.id
                    ready_q.pop(0)

        flag = True
        for p in new_list:
            if stat[ord(p.id) - 65] != "Term":
                flag = False
                break
        if flag:
            total_burst = 0
            for p in new_list:
                total_burst += p.num_burst

            for i in range(len(end)):
                for j in range(len(end[i])):
                    tt_t += end[i][j] - arrive[i][j] + switch / 2

            avg_burst = burst_t / total_burst
            avg_tt = tt_t / total_burst
            avg_wait = wait_t / total_burst

            print("time {}ms: Simulator ended for RR [Q <empty>]".format(time))
            return [avg_burst, avg_wait, avg_tt, num_switch, num_prem]

        wait_t += len(ready_q)
        time += 1


if __name__ == "__main__":
    rand_seed = int(sys.argv[1])
    lam = float(sys.argv[2])
    exp_bound = int(sys.argv[3])
    num_process = int(sys.argv[4])
    switch_time = int(sys.argv[5])
    alpha = float(sys.argv[6])
    time_slice = int(sys.argv[7])

    RR_flag = False
    if len(sys.argv) > 8:
        RR_flag = sys.argv[8]
        if RR_flag == "BEGINNING":
            RR_flag = True
        elif RR_flag == "END":
            RR_flag = False

    rand_obj.srand(rand_seed)

    # Random Init
    rand_config(lam, exp_bound, num_process, alpha)
    # process_list.sort(key=lambda x: x.arrive_time)

    FCFS_result = FCFS(switch_time)
    print_time("FCFS", FCFS_result)
    print()
    sys.stderr.write("\n")

    SJF_result = SJF(switch_time)
    print_time("SJF", SJF_result)
    print()
    sys.stderr.write("\n")

    SRT_list = SRT(num_process, switch_time)

    total_turnaround = 0
    for p in SRT_list[1]:
        for t in p:
            total_turnaround += t[1] - t[0]

    print_time("SRT", [SRT_list[0], SRT_list[2] / SRT_list[3], total_turnaround/SRT_list[3], SRT_list[5], SRT_list[4]])

    print()
    sys.stderr.write("\n")

    RR_list = RR(time_slice, switch_time, RR_flag)
    print_time("RR", RR_list)
