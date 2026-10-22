"""Tiny in-process task queue: workers, retries, dead-letter list. Stdlib only.

Teaches the ideas behind Celery/RabbitMQ. Not a replacement for them.
"""
import queue
import threading
import time


class TaskQueue:
    def __init__(self, workers=2, max_retries=2, backoff=0.0, sleep=time.sleep):
        self.max_retries = max_retries
        self.backoff = backoff
        self.sleep = sleep
        self.n_workers = workers
        self._q = queue.Queue()
        self._threads = []
        self._lock = threading.Lock()
        self.results = {}       # task_id -> result
        self.dead_letters = []  # (task_id, name, last_error)
        self._next_id = 0
        self._tasks = {}

    def task(self, fn):
        self._tasks[fn.__name__] = fn
        return fn

    def submit(self, name, *args, **kwargs):
        with self._lock:
            self._next_id += 1
            task_id = self._next_id
        self._q.put((task_id, name, args, kwargs, 0))
        return task_id

    def start(self):
        for _ in range(self.n_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self._threads.append(t)

    def _worker(self):
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            task_id, name, args, kwargs, attempt = item
            try:
                result = self._tasks[name](*args, **kwargs)
                with self._lock:
                    self.results[task_id] = result
            except Exception as exc:
                if attempt < self.max_retries:
                    self.sleep(self.backoff * (2 ** attempt))
                    self._q.put((task_id, name, args, kwargs, attempt + 1))
                else:
                    with self._lock:
                        self.dead_letters.append((task_id, name, repr(exc)))
            finally:
                self._q.task_done()

    def join(self):
        """Wait until every submitted task (including retries) is finished."""
        self._q.join()

    def stop(self):
        for _ in self._threads:
            self._q.put(None)
        for t in self._threads:
            t.join()
        self._threads = []


if __name__ == "__main__":
    q = TaskQueue(workers=2, max_retries=1)

    @q.task
    def embed(doc):
        if doc == "bad":
            raise ValueError("cannot parse")
        return len(doc)

    q.start()
    for d in ["hello", "bad", "world!"]:
        q.submit("embed", d)
    q.join()
    q.stop()
    print(q.results, q.dead_letters)
