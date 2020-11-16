from _03_bridge.base_bridge import BridgeBase

from datetime import timedelta, datetime
import time

from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler

from ptan.ignite import EndOfEpisodeHandler, EpisodeEvents, PeriodicEvents, PeriodEvents


class TimeHandler:
    TIME_PASSED_METRIC = 'time_passed'

    def __init__(self):
        self._started_ts = time.time()

    def attach(self, engine: Engine):
        engine.add_event_handler(EpisodeEvents.EPISODE_COMPLETED, self)

    def __call__(self, engine: Engine):
        engine.state.metrics[self.TIME_PASSED_METRIC] = time.time() - self._started_ts

class LoopControlV10:
    def __init__(self, bridge:BridgeBase, run_name:str, bound_avg_reward:float=1000.0, logtb:bool = False, logfolder:str = "runs"):

        self.logfolder = logfolder
        self.bridge = bridge
        self.run_name = run_name
        self.engine = Engine(self.bridge.process_batch)

        # this handler has several problems. it does more than one thing and it also
        # has to have direct access to the experienceSource of the agent.
        # that could be refactored
        EndOfEpisodeHandler(self.bridge.agent.exp_source, bound_avg_reward = bound_avg_reward).attach(self.engine)
        TimeHandler().attach(self.engine)

        RunningAverage(output_transform=lambda v: v['loss']).attach(self.engine, "avg_loss")
        PeriodicEvents().attach(self.engine) # creates periodic events

        @self.engine.on(EpisodeEvents.EPISODE_COMPLETED)
        def episode_completed(trainer: Engine):
            passed = trainer.state.metrics.get('time_passed', 0)
            print("Episode %d: reward=%.0f, steps=%s, "
                  "elapsed=%s" % (
                      trainer.state.episode, trainer.state.episode_reward,
                      trainer.state.episode_steps,
                      timedelta(seconds=int(passed))))

        @self.engine.on(EpisodeEvents.BOUND_REWARD_REACHED)
        def game_solved(trainer: Engine):
            passed = trainer.state.metrics['time_passed']
            print("Game solved in %s, after %d episodes "
                  "and %d iterations!" % (
                      timedelta(seconds=int(passed)),
                      trainer.state.episode, trainer.state.iteration))
            trainer.should_terminate = True
        if logtb:
            tb = self._create_tb_logger()
            handler = OutputHandler(tag="episodes", metric_names=['reward', 'steps', 'avg_reward'])
            tb.attach(self.engine, log_handler=handler, event_name=EpisodeEvents.EPISODE_COMPLETED)

            handler = OutputHandler(tag="train", metric_names=['avg_loss'], output_transform=lambda a: a)
            tb.attach(self.engine, log_handler=handler, event_name=PeriodEvents.ITERS_100_COMPLETED)


    def _create_tb_logger(self) -> TensorboardLogger:
        now = datetime.now().isoformat(timespec='minutes')
        now = now.replace(":", "")
        logdir = f"{self.logfolder}/{now}-{self.run_name}"
        return TensorboardLogger(log_dir=logdir)


    def run(self):
        self.engine.run(self.bridge.batch_generator())