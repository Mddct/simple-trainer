# modified from https://github.com/google/CommonLoopUtils
import abc
import collections
import concurrent.futures
import contextlib
import functools
import os
import sys
import threading
from collections.abc import Mapping, Sequence
from typing import Any, Callable, List, Optional, Union

import numpy as np
import tensorboard
import wrapt
from absl import flags, logging
from etils import epath

Array = Union[np.ndarray]
Scalar = Union[int, float, np.number, np.ndarray]


class MetricWriter(abc.ABC):
    """MetricWriter inferface."""

    @abc.abstractmethod
    def write_summaries(self,
                        step: int,
                        values: Mapping[str, Array],
                        metadata: Optional[Mapping[str, Any]] = None):
        """Saves an arbitrary tensor summary.

    Useful when working with custom plugins or constructing a summary directly.

    Args:
      step: Step at which the scalar values occurred.
      values: Mapping from tensor keys to tensors.
      metadata: Optional SummaryMetadata, as a proto or serialized bytes.
                Note that markdown formatting is rendered by tensorboard.
    """

    @abc.abstractmethod
    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        """Write scalar values for the step.

    Consecutive calls to this method can provide different sets of scalars.
    Repeated writes for the same metric at the same step are not allowed.

    Args:
      step: Step at which the scalar values occurred.
      scalars: Mapping from metric name to value.
    """

    @abc.abstractmethod
    def write_images(self, step: int, images: Mapping[str, Array]):
        """Write images for the step.

    Consecutive calls to this method can provide different sets of images.
    Repeated writes for the same image key at the same step are not allowed.

    Warning: Not all MetricWriter implementation support writing images!

    Args:
      step: Step at which the images occurred.
      images: Mapping from image key to images. Images should have the shape [N,
        H, W, C] or [H, W, C], where H is the height, W is the width and C the
        number of channels (1 or 3). N is the number of images that will be
        written. Image dimensions can differ between different image keys but
        not between different steps for the same image key.
    """

    @abc.abstractmethod
    def write_videos(self, step: int, videos: Mapping[str, Array]):
        """Write videos for the step.

    Warning: Logging only.
    Not all MetricWriter implementation support writing videos!

    Consecutive calls to this method can provide different sets of videos.
    Repeated writes for the same video key at the same step are not allowed.


    Args:
      step: Step at which the videos occurred.
      videos: Mapping from video key to videos. videos should have the shape
        [N, T, H, W, C] or [T, H, W, C], where T is time, H is the height,
        W is the width and C the number of channels (1 or 3). N is the number
        of videos that will be written. Video dimensions can differ between
        different video keys but not between different steps for the same
        video key.
    """

    @abc.abstractmethod
    def write_audios(self, step: int, audios: Mapping[str, Array], *,
                     sample_rate: int):
        """Write audios for the step.

    Consecutive calls to this method can provide different sets of audios.
    Repeated writes for the same audio key at the same step are not allowed.

    Warning: Not all MetricWriter implementation support writing audios!

    Args:
      step: Step at which the audios occurred.
      audios: Mapping from audio key to audios. Audios should have the shape
        [N, T, C], where T is the time length and C the number of channels
        (1 - mono, 2 - stereo, >= 3 - surround; not all writers support any
        number of channels). N is the number of audios that will be written.
        Audio dimensions can differ between different audio keys but not between
        different steps for the same audio key. Values should be floating-point
        values in [-1, +1].
      sample_rate: Sample rate for the audios.
    """

    @abc.abstractmethod
    def write_texts(self, step: int, texts: Mapping[str, str]):
        """Writes text snippets for the step.

    Warning: Not all MetricWriter implementation support writing text!

    Args:
      step: Step at which the text snippets occurred.
      texts: Mapping from name to text snippet.
    """

    @abc.abstractmethod
    def write_histograms(self,
                         step: int,
                         arrays: Mapping[str, Array],
                         num_buckets: Optional[Mapping[str, int]] = None):
        """Writes histograms for the step.

    Consecutive calls to this method can provide different sets of scalars.
    Repeated writes for the same metric at the same step are not allowed.

    Warning: Not all MetricWriter implementation support writing histograms!

    Args:
      step: Step at which the arrays were generated.
      arrays: Mapping from name to arrays to summarize.
      num_buckets: Number of buckets used to create the histogram of the arrays.
        The default number of buckets depends on the particular implementation
        of the MetricWriter.
    """

    def write_pointcloud(
        self,
        step: int,
        point_clouds: Mapping[str, Array],
        *,
        point_colors: Mapping[str, Array] | None = None,
        configs: Mapping[str, str | float | bool | None] | None = None,
    ):
        """Writes point cloud summaries.

    Args:
      step: Step at which the point cloud was generated.
      point_clouds: Mapping from point clouds key to point cloud of shape [N, 3]
        array of point coordinates.
      point_colors: Mapping from point colors key to [N, 3] array of point
        colors.
      configs: A dictionary of configuration options for the point cloud.
    """
        raise NotImplementedError()

    @abc.abstractmethod
    def write_hparams(self, hparams: Mapping[str, Any]):
        """Write hyper parameters.

    Do not call twice.

    Args:
      hparams: Flat mapping from hyper parameter name to value.
    """

    @abc.abstractmethod
    def flush(self):
        """Tells the MetricWriter to write out any cached values."""

    @abc.abstractmethod
    def close(self):
        """Flushes and closes the MetricWriter.

    Calling any method on MetricWriter after MetricWriter.close()
    is undefined behavior.
    """


class LoggingWriter(MetricWriter):
    """MetricWriter that writes all values to INFO log."""

    def __init__(self, collection: Optional[str] = None):
        if collection:
            self._collection_str = f" collection={collection}"
        else:
            self._collection_str = ""

    def write_summaries(self,
                        step: int,
                        values: Mapping[str, Array],
                        metadata: Optional[Mapping[str, Any]] = None):
        logging.info("[%d]%s Got raw tensors: %s.", step, self._collection_str,
                     {
                         k: v.shape
                         for k, v in values.items()
                     })

    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        values = [
            f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in sorted(scalars.items())
        ]
        logging.info("[%d]%s %s", step, self._collection_str,
                     ", ".join(values))

    def write_images(self, step: int, images: Mapping[str, Array]):
        logging.info("[%d]%s Got images: %s.", step, self._collection_str, {
            k: v.shape
            for k, v in images.items()
        })

    def write_videos(self, step: int, videos: Mapping[str, Array]):
        logging.info("[%d]%s Got videos: %s.", step, self._collection_str, {
            k: v.shape
            for k, v in videos.items()
        })

    def write_audios(self, step: int, audios: Mapping[str, Array], *,
                     sample_rate: int):
        logging.info("[%d]%s Got audios: %s.", step, self._collection_str, {
            k: v.shape
            for k, v in audios.items()
        })

    def write_texts(self, step: int, texts: Mapping[str, str]):
        logging.info("[%d]%s Got texts: %s.", step, self._collection_str,
                     texts)

    def write_histograms(self,
                         step: int,
                         arrays: Mapping[str, Array],
                         num_buckets: Optional[Mapping[str, int]] = None):
        pass

    def write_pointcloud(
        self,
        step: int,
        point_clouds: Mapping[str, Array],
        *,
        point_colors: Mapping[str, Any] | None = None,
        configs: Mapping[str, str | float | bool | None] | None = None,
    ):
        logging.info(
            "[%d]%s Got point clouds: %s, point_colors: %s, configs: %s.",
            step,
            self._collection_str,
            {
                k: v.shape
                for k, v in point_clouds.items()
            },
            ({
                k: v.shape
                for k, v in point_colors.items()
            } if point_colors is not None else None),
            configs,
        )

    def write_hparams(self, hparams: Mapping[str, Any]):
        logging.info("[Hyperparameters]%s %s", self._collection_str, hparams)

    def flush(self):
        logging.flush()

    def close(self):
        self.flush()


class TorchTensorboardWriter(MetricWriter):
    """MetricWriter that writes Pytorch summary files."""

    def __init__(self, logdir: str):
        super().__init__()
        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=logdir)

    def write_summaries(self,
                        step: int,
                        values: Mapping[str, Array],
                        metadata: Optional[Mapping[str, Any]] = None):
        logging.log_first_n(
            logging.WARNING,
            "TorchTensorboardWriter does not support writing raw summaries.",
            1)

    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        for key, value in scalars.items():
            self._writer.add_scalar(key, value, global_step=step)

    def write_images(self, step: int, images: Mapping[str, Array]):
        for key, value in images.items():
            self._writer.add_image(key,
                                   value,
                                   global_step=step,
                                   dataformats="HWC")

    def write_videos(self, step: int, videos: Mapping[str, Array]):
        logging.log_first_n(
            logging.WARNING,
            "TorchTensorBoardWriter does not support writing videos.", 1)

    def write_audios(self, step: int, audios: Mapping[str, Array], *,
                     sample_rate: int):
        for key, value in audios.items():
            self._writer.add_audio(key,
                                   value,
                                   global_step=step,
                                   sample_rate=sample_rate)

    def write_texts(self, step: int, texts: Mapping[str, str]):
        raise NotImplementedError(
            "TorchTensorBoardWriter does not support writing texts.")

    def write_histograms(self,
                         step: int,
                         arrays: Mapping[str, Array],
                         num_buckets: Optional[Mapping[str, int]] = None):
        for tag, values in arrays.items():
            bins = None if num_buckets is None else num_buckets.get(tag)
            self._writer.add_histogram(tag,
                                       values,
                                       global_step=step,
                                       bins="auto",
                                       max_bins=bins)

    def write_pointcloud(
        self,
        step: int,
        point_clouds: Mapping[str, Array],
        *,
        point_colors: Mapping[str, Array] | None = None,
        configs: Mapping[str, str | float | bool | None] | None = None,
    ):
        logging.log_first_n(
            logging.WARNING,
            "TorchTensorBoardWriter does not support writing point clouds.",
            1,
        )

    def write_hparams(self, hparams: Mapping[str, Any]):
        self._writer.add_hparams(hparams, {})

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()


class MultiWriter(MetricWriter):
    """MetricWriter that writes to multiple writers at once."""

    def __init__(self, writers: Sequence[MetricWriter]):
        self._writers = tuple(writers)

    def write_summaries(self,
                        step: int,
                        values: Mapping[str, Array],
                        metadata: Optional[Mapping[str, Any]] = None):
        for w in self._writers:
            w.write_summaries(step, values, metadata)

    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        for w in self._writers:
            w.write_scalars(step, scalars)

    def write_images(self, step: int, images: Mapping[str, Array]):
        for w in self._writers:
            w.write_images(step, images)

    def write_videos(self, step: int, videos: Mapping[str, Array]):
        for w in self._writers:
            w.write_videos(step, videos)

    def write_audios(self, step: int, audios: Mapping[str, Array], *,
                     sample_rate: int):
        for w in self._writers:
            w.write_audios(step, audios, sample_rate=sample_rate)

    def write_texts(self, step: int, texts: Mapping[str, str]):
        for w in self._writers:
            w.write_texts(step, texts)

    def write_histograms(self,
                         step: int,
                         arrays: Mapping[str, Array],
                         num_buckets: Optional[Mapping[str, int]] = None):
        for w in self._writers:
            w.write_histograms(step, arrays, num_buckets)

    def write_pointcloud(
        self,
        step: int,
        point_clouds: Mapping[str, Array],
        *,
        point_colors: Mapping[str, Array] | None = None,
        configs: Mapping[str, str | float | bool | None] | None = None,
    ):
        for w in self._writers:
            w.write_pointcloud(step,
                               point_clouds,
                               point_colors=point_colors,
                               configs=configs)

    def write_hparams(self, hparams: Mapping[str, Any]):
        for w in self._writers:
            w.write_hparams(hparams)

    def flush(self):
        for w in self._writers:
            w.flush()

    def close(self):
        for w in self._writers:
            w.close()


class AsyncError(Exception):
    """An exception that wraps another exception that ocurred asynchronously."""


class Pool:
    """Pool for wrapping functions to be executed asynchronously.

  Synopsis:

    from clu.internal import asynclib

    pool = asynclib.Pool()
    @pool
    def fn():
      time.sleep(1)

    future = fn()
    print(future.result())
    fn()  # This could re-raise an exception from the first execution.
    print(len(pool))  # Would print "1" because there is one function in flight.
    pool.flush()  # This could re-raise an exception from the second execution.
  """

    def __init__(self,
                 thread_name_prefix: str = "",
                 max_workers: Optional[int] = None):
        """Creates a new pool that decorates functions for async execution.

    Args:
      thread_name_prefix: See documentation of `ThreadPoolExecutor`.
      max_workers: See documentation of `ThreadPoolExecutor`. The default `None`
        optimizes for parallelizability using the number of CPU cores. If you
        specify `max_workers=1` you the async calls are executed in the same
        order they have been scheduled.
    """
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix=thread_name_prefix)
        self._max_workers = max_workers
        self._thread_name_prefix = thread_name_prefix
        self._errors = collections.deque()
        self._errors_mutex = threading.Lock()
        self._queue_length = 0

    def _reraise(self) -> None:
        if self._errors:
            with self._errors_mutex:
                exc_info = self._errors.popleft()
            exc = exc_info[1].with_traceback(exc_info[2])
            raise AsyncError(
                f"Error '{exc}' occurred ASYNCHRONOUSLY.") from exc

    def close(self) -> None:
        """Closes this pool & raise a pending exception (if needed)."""
        self._pool.shutdown(wait=True)
        self._reraise()

    def join(self) -> None:
        """Blocks until all functions are processed.

    The pool can be used to schedule more functions after calling this function,
    but there might be more exceptions

    Side-effect:
      If any of the functions raised an exception, then the first of these
      exceptions is reraised.
    """
        self._pool.shutdown(wait=True)
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix=self._thread_name_prefix)
        self._reraise()

    @property
    def queue_length(self) -> int:
        """Returns the number of functions that have not returned yet."""
        return self._queue_length

    @property
    def has_errors(self) -> bool:
        """Returns True if there are any pending errors."""
        return bool(self._errors)

    def clear_errors(self) -> List[Exception]:
        """Clears all pending errors and returns them as a (possibly empty) list."""
        with self._errors_mutex:
            errors, self._errors = self._errors, collections.deque()
        return list(errors)

    def __call__(self, fn: Callable):  # pylint: disable=g-bare-generic
        """Returns an async version of fn.

    The function will be executed by this class's ThreadPoolExecutor. Any errors
    will be stored and re-raised next time any function is called that is
    executed through this pool.

    Note that even if there was a previous error, the function is still
    scheduled upon re-execution of the wrapper returned by this function.

    Args:
      fn: Function to be wrapped.

    Returns:
      An async version of `fn`. The return value of that async version will be
      a future (unless an exception was re-raised).
    """

        def inner(*args, **kwargs):

            def trap_errors(*args, **kwargs):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    with self._errors_mutex:
                        self._errors.append(sys.exc_info())
                    logging.exception("Error in producer thread for %s",
                                      self._thread_name_prefix)
                    raise e
                finally:
                    self._queue_length -= 1

            self._queue_length += 1
            if not self.has_errors:
                return self._pool.submit(trap_errors, *args, **kwargs)
            self._pool.submit(trap_errors, *args, **kwargs)
            self._reraise()

        if isinstance(fn.__name__, str):
            # Regular function.
            return functools.wraps(fn)(inner)
        # Mock or another weird function that fails with functools.wraps().
        return inner


@wrapt.decorator
def _wrap_exceptions(wrapped, instance, args, kwargs):
    del instance
    try:
        return wrapped(*args, **kwargs)
    except asynclib.AsyncError as e:
        raise asynclib.AsyncError(
            "Consider re-running the code without AsyncWriter (e.g. creating a "
            "writer using "
            "`clu.metric_writers.create_default_writer(asynchronous=False)`)"
        ) from e


class AsyncWriter(MetricWriter):
    """MetricWriter that performs write operations in a separate thread.

  All write operations will be executed in a background thread. If an exceptions
  occurs in the background thread it will be raised on the main thread on the
  call of one of the write_* methods.

  Use num_workers > 1 at your own risk, if the underlying writer is not
  thread-safe or does not expect out-of-order events, this can cause problems.
  If num_workers is None then the ThreadPool will use `os.cpu_count()`
  processes.
  """

    def __init__(self,
                 writer: MetricWriter,
                 *,
                 num_workers: Optional[int] = 1):
        super().__init__()
        self._writer = writer
        # By default, we have a thread pool with a single worker to ensure that
        # calls to the function are run in order (but in a background thread).
        self._num_workers = num_workers
        self._pool = Pool(thread_name_prefix="AsyncWriter",
                          max_workers=num_workers)

    @_wrap_exceptions
    def write_summaries(self,
                        step: int,
                        values: Mapping[str, Array],
                        metadata: Optional[Mapping[str, Any]] = None):
        self._pool(self._writer.write_summaries)(step=step,
                                                 values=values,
                                                 metadata=metadata)

    @_wrap_exceptions
    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        self._pool(self._writer.write_scalars)(step=step, scalars=scalars)

    @_wrap_exceptions
    def write_images(self, step: int, images: Mapping[str, Array]):
        self._pool(self._writer.write_images)(step=step, images=images)

    @_wrap_exceptions
    def write_videos(self, step: int, videos: Mapping[str, Array]):
        self._pool(self._writer.write_videos)(step=step, videos=videos)

    @_wrap_exceptions
    def write_audios(self, step: int, audios: Mapping[str, Array], *,
                     sample_rate: int):
        self._pool(self._writer.write_audios)(step=step,
                                              audios=audios,
                                              sample_rate=sample_rate)

    @_wrap_exceptions
    def write_texts(self, step: int, texts: Mapping[str, str]):
        self._pool(self._writer.write_texts)(step=step, texts=texts)

    @_wrap_exceptions
    def write_histograms(self,
                         step: int,
                         arrays: Mapping[str, Array],
                         num_buckets: Optional[Mapping[str, int]] = None):
        self._pool(self._writer.write_histograms)(step=step,
                                                  arrays=arrays,
                                                  num_buckets=num_buckets)

    @_wrap_exceptions
    def write_pointcloud(
        self,
        step: int,
        point_clouds: Mapping[str, Array],
        *,
        point_colors: Mapping[str, Array] | None = None,
        configs: Mapping[str, str | float | bool | None] | None = None,
    ):
        self._pool(self._writer.write_pointcloud)(
            step=step,
            point_clouds=point_clouds,
            point_colors=point_colors,
            configs=configs,
        )

    @_wrap_exceptions
    def write_hparams(self, hparams: Mapping[str, Any]):
        self._pool(self._writer.write_hparams)(hparams=hparams)

    def flush(self):
        try:
            self._pool.join()
        finally:
            self._writer.flush()

    def close(self):
        try:
            self.flush()
        finally:
            self._writer.close()


class AsyncMultiWriter(MultiWriter):
    """AsyncMultiWriter writes to multiple writes in a separate thread."""

    def __init__(self,
                 writers: Sequence[MetricWriter],
                 *,
                 num_workers: Optional[int] = 1):
        super().__init__(
            [AsyncWriter(w, num_workers=num_workers) for w in writers])


@contextlib.contextmanager
def ensure_flushes(*writers: MetricWriter):
    """Context manager which ensures that one or more writers are flushed."""
    try:
        # The caller should not need to use the yielded value, but we yield
        # the first writer to stay backwards compatible for a single writer.
        yield writers[0]
    finally:
        for writer in writers:
            writer.flush()


def _is_scalar(value: Any) -> bool:
    if isinstance(value, values.Scalar) or isinstance(value,
                                                      (int, float, np.number)):
        return True
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return value.ndim == 0 or value.size <= 1
    return False


def create_default_writer(
    logdir: Optional[epath.PathLike] = None,
    *,
    just_logging: bool = False,
    asynchronous: bool = True,
    collection: Optional[str] = None,
) -> MultiWriter:
    """Create the default writer for the platform.

    On most platforms this will create a MultiWriter that writes to multiple back
    ends (logging, TF summaries etc.).

    Args:
      logdir: Logging dir to use for torch summary files. If empty/None will the
        returned writer will not write torch summary files.
      just_logging: If True only use a LoggingWriter. This is useful in multi-host
        setups when only the first host should write metrics and all other hosts
        should only write to their own logs.
        default (None) will automatically determine if you # GOOGLE-INTERNAL have
      asynchronous: If True return an AsyncMultiWriter to not block when writing
        metrics.
      collection: A string which, if provided, provides an indication that the
        provided metrics should all be written to the same collection, or
        grouping.

    Returns:
      A `MetricWriter` according to the platform and arguments.
    """
    if just_logging:
        if asynchronous:
            return AsyncMultiWriter([LoggingWriter(collection=collection)])
        else:
            return MultiWriter([LoggingWriter(collection=collection)])
    writers = [LoggingWriter(collection=collection)]
    if logdir is not None:
        logdir = epath.Path(logdir)
        if collection is not None:
            logdir /= collection
        writers.append(TorchTensorboardWriter(os.fspath(logdir)))
    if asynchronous:
        return AsyncMultiWriter(writers)
    return MultiWriter(writers)
