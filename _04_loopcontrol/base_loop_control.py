from abc import ABC, abstractmethod
from _03_bridge.base_bridge import BridgeBase

import time
from datetime import timedelta, datetime

from ptan.ignite import EndOfEpisodeHandler, EpisodeEvents, PeriodicEvents, PeriodEvents
from ignite.engine import Engine

from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler

class TimeHandler:
    TIME_PASSED_METRIC = 'time_passed'

    def __init__(self):
        self._started_ts = time.time()

    def attach(self, engine: Engine):
        engine.add_event_handler(EpisodeEvents.EPISODE_COMPLETED, self)

    def __call__(self, engine: Engine):
        engine.state.metrics[self.TIME_PASSED_METRIC] = time.time() - self._started_ts

class LoopControlBase(ABC):

    def __init__(self, bridge:BridgeBase, run_name:str, bound_avg_reward:float=1000.0, logtb:bool = False,
                 logfolder:str = "runs"):

        self.logfolder = logfolder
        self.bridge = bridge
        self.run_name = run_name
        self.bound_avg_reward = bound_avg_reward
        self.logtb = logtb

        self.engine = Engine(self.bridge.process_batch)

        self.tblogger = self._create_tb_logger()


    def _create_tb_logger(self) -> TensorboardLogger:
        now = datetime.now().isoformat(timespec='minutes')
        now = now.replace(":", "")
        logdir = f"{self.logfolder}/{now}-{self.run_name}"
        return TensorboardLogger(log_dir=logdir)

    def episode_completed_basic(self,trainer: Engine):
        passed = trainer.state.metrics.get('time_passed', 0)
        print("Episode %d: reward=%.0f, steps=%s, "
              "elapsed=%s" % (
                  trainer.state.episode, trainer.state.episode_reward,
                  trainer.state.episode_steps,
                  timedelta(seconds=int(passed))))

    def game_solved_basic(self,trainer: Engine):
        passed = trainer.state.metrics['time_passed']
        print("Game solved in %s, after %d episodes "
              "and %d iterations!" % (
                  timedelta(seconds=int(passed)),
                  trainer.state.episode, trainer.state.iteration))
        trainer.should_terminate = True

    def run(self):
        self.engine.run(self.bridge.batch_generator())