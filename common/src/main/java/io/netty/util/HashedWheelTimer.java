/*
 * Copyright 2012 The Netty Project
 *
 * The Netty Project licenses this file to you under the Apache License,
 * version 2.0 (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at:
 *
 *   https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package io.netty.util;

import static io.netty.util.internal.ObjectUtil.checkInRange;
import static io.netty.util.internal.ObjectUtil.checkPositive;
import static io.netty.util.internal.ObjectUtil.checkNotNull;

import io.netty.util.concurrent.ImmediateExecutor;
import io.netty.util.internal.PlatformDependent;
import io.netty.util.internal.logging.InternalLogger;
import io.netty.util.internal.logging.InternalLoggerFactory;

import java.util.Collections;
import java.util.HashSet;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerFieldUpdater;
import java.util.concurrent.atomic.AtomicLong;

import static io.netty.util.internal.StringUtil.simpleClassName;

/**
 * A {@link Timer} optimized for approximated I/O timeout scheduling.
 *
 * <h3>Tick Duration</h3>
 *
 * As described with 'approximated', this timer does not execute the scheduled
 * {@link TimerTask} on time.  {@link HashedWheelTimer}, on every tick, will
 * check if there are any {@link TimerTask}s behind the schedule and execute
 * them.
 * <p>
 * You can increase or decrease the accuracy of the execution timing by
 * specifying smaller or larger tick duration in the constructor.  In most
 * network applications, I/O timeout does not need to be accurate.  Therefore,
 * the default tick duration is 100 milliseconds and you will not need to try
 * different configurations in most cases.
 *
 * <h3>Ticks per Wheel (Wheel Size)</h3>
 *
 * {@link HashedWheelTimer} maintains a data structure called 'wheel'.
 * To put simply, a wheel is a hash table of {@link TimerTask}s whose hash
 * function is 'dead line of the task'.  The default number of ticks per wheel
 * (i.e. the size of the wheel) is 512.  You could specify a larger value
 * if you are going to schedule a lot of timeouts.
 *
 * <h3>Do not create many instances.</h3>
 *
 * {@link HashedWheelTimer} creates a new thread whenever it is instantiated and
 * started.  Therefore, you should make sure to create only one instance and
 * share it across your application.  One of the common mistakes, that makes
 * your application unresponsive, is to create a new instance for every connection.
 *
 * <h3>Implementation Details</h3>
 *
 * {@link HashedWheelTimer} is based on
 * <a href="https://cseweb.ucsd.edu/users/varghese/">George Varghese</a> and
 * Tony Lauck's paper,
 * <a href="https://cseweb.ucsd.edu/users/varghese/PAPERS/twheel.ps.Z">'Hashed
 * and Hierarchical Timing Wheels: data structures to efficiently implement a
 * timer facility'</a>.  More comprehensive slides are located
 * <a href="https://www.cse.wustl.edu/~cdgill/courses/cs6874/TimingWheels.ppt">here</a>.
 */
public class HashedWheelTimer implements Timer {

    static final InternalLogger logger =
            InternalLoggerFactory.getInstance(HashedWheelTimer.class);

    private static final AtomicInteger INSTANCE_COUNTER = new AtomicInteger();
    private static final AtomicBoolean WARNED_TOO_MANY_INSTANCES = new AtomicBoolean();
    private static final int INSTANCE_COUNT_LIMIT = 64;
    private static final long MILLISECOND_NANOS = TimeUnit.MILLISECONDS.toNanos(1);
    private static final ResourceLeakDetector<HashedWheelTimer> leakDetector = ResourceLeakDetectorFactory.instance()
            .newResourceLeakDetector(HashedWheelTimer.class, 1);

    private static final AtomicIntegerFieldUpdater<HashedWheelTimer> WORKER_STATE_UPDATER =
            AtomicIntegerFieldUpdater.newUpdater(HashedWheelTimer.class, "workerState");

    private final ResourceLeakTracker<HashedWheelTimer> leak;
    private final Worker worker = new Worker();  // 最主要的一个线程，负责进行拨动时钟，并调用相应的任务
    private final Thread workerThread;  // 就是执行前一行的worker

    public static final int WORKER_STATE_INIT = 0;
    public static final int WORKER_STATE_STARTED = 1;
    public static final int WORKER_STATE_SHUTDOWN = 2;
    @SuppressWarnings({"unused", "FieldMayBeFinal"})
    private volatile int workerState; // 0 - init, 1 - started, 2 - shut down

    private final long tickDuration;
    private final HashedWheelBucket[] wheel;  // 时间轮
    private final int mask;
    private final CountDownLatch startTimeInitialized = new CountDownLatch(1);
    private final Queue<HashedWheelTimeout> timeouts = PlatformDependent.newMpscQueue();  // 存储所有task
    private final Queue<HashedWheelTimeout> cancelledTimeouts = PlatformDependent.newMpscQueue();
    private final AtomicLong pendingTimeouts = new AtomicLong(0);
    private final long maxPendingTimeouts;
    private final Executor taskExecutor;

    private volatile long startTime;

    /**
     * Creates a new timer with the default thread factory
     * ({@link Executors#defaultThreadFactory()}), default tick duration, and
     * default number of ticks per wheel.
     */
    public HashedWheelTimer() {
        this(Executors.defaultThreadFactory());
    }

    /**
     * Creates a new timer with the default thread factory
     * ({@link Executors#defaultThreadFactory()}) and default number of ticks
     * per wheel.
     *
     * @param tickDuration the duration between tick
     * @param unit         the time unit of the {@code tickDuration}
     * @throws NullPointerException     if {@code unit} is {@code null}
     * @throws IllegalArgumentException if {@code tickDuration} is &lt;= 0
     */
    public HashedWheelTimer(long tickDuration, TimeUnit unit) {
        this(Executors.defaultThreadFactory(), tickDuration, unit);
    }

    /**
     * Creates a new timer with the default thread factory
     * ({@link Executors#defaultThreadFactory()}).
     *
     * @param tickDuration  the duration between tick
     * @param unit          the time unit of the {@code tickDuration}
     * @param ticksPerWheel the size of the wheel
     * @throws NullPointerException     if {@code unit} is {@code null}
     * @throws IllegalArgumentException if either of {@code tickDuration} and {@code ticksPerWheel} is &lt;= 0
     */
    public HashedWheelTimer(long tickDuration, TimeUnit unit, int ticksPerWheel) {
        this(Executors.defaultThreadFactory(), tickDuration, unit, ticksPerWheel);
    }

    /**
     * Creates a new timer with the default tick duration and default number of
     * ticks per wheel.
     *
     * @param threadFactory a {@link ThreadFactory} that creates a
     *                      background {@link Thread} which is dedicated to
     *                      {@link TimerTask} execution.
     * @throws NullPointerException if {@code threadFactory} is {@code null}
     */
    public HashedWheelTimer(ThreadFactory threadFactory) {
        this(threadFactory, 100, TimeUnit.MILLISECONDS);
    }

    /**
     * Creates a new timer with the default number of ticks per wheel.
     *
     * @param threadFactory a {@link ThreadFactory} that creates a
     *                      background {@link Thread} which is dedicated to
     *                      {@link TimerTask} execution.
     * @param tickDuration  the duration between tick
     * @param unit          the time unit of the {@code tickDuration}
     * @throws NullPointerException     if either of {@code threadFactory} and {@code unit} is {@code null}
     * @throws IllegalArgumentException if {@code tickDuration} is &lt;= 0
     */
    public HashedWheelTimer(
            ThreadFactory threadFactory, long tickDuration, TimeUnit unit) {
        this(threadFactory, tickDuration, unit, 512);
    }

    /**
     * Creates a new timer.
     *
     * @param threadFactory a {@link ThreadFactory} that creates a
     *                      background {@link Thread} which is dedicated to
     *                      {@link TimerTask} execution.
     * @param tickDuration  the duration between tick
     * @param unit          the time unit of the {@code tickDuration}
     * @param ticksPerWheel the size of the wheel
     * @throws NullPointerException     if either of {@code threadFactory} and {@code unit} is {@code null}
     * @throws IllegalArgumentException if either of {@code tickDuration} and {@code ticksPerWheel} is &lt;= 0
     */
    public HashedWheelTimer(
            ThreadFactory threadFactory,
            long tickDuration, TimeUnit unit, int ticksPerWheel) {
        this(threadFactory, tickDuration, unit, ticksPerWheel, true);
    }

    /**
     * Creates a new timer.
     *
     * @param threadFactory a {@link ThreadFactory} that creates a
     *                      background {@link Thread} which is dedicated to
     *                      {@link TimerTask} execution.
     * @param tickDuration  the duration between tick
     * @param unit          the time unit of the {@code tickDuration}
     * @param ticksPerWheel the size of the wheel
     * @param leakDetection {@code true} if leak detection should be enabled always,
     *                      if false it will only be enabled if the worker thread is not
     *                      a daemon thread.
     * @throws NullPointerException     if either of {@code threadFactory} and {@code unit} is {@code null}
     * @throws IllegalArgumentException if either of {@code tickDuration} and {@code ticksPerWheel} is &lt;= 0
     */
    public HashedWheelTimer(
            ThreadFactory threadFactory,
            long tickDuration, TimeUnit unit, int ticksPerWheel, boolean leakDetection) {
        this(threadFactory, tickDuration, unit, ticksPerWheel, leakDetection, -1);
    }

    /**
     * Creates a new timer.
     *
     * @param threadFactory        a {@link ThreadFactory} that creates a
     *                             background {@link Thread} which is dedicated to
     *                             {@link TimerTask} execution.
     * @param tickDuration         the duration between tick
     * @param unit                 the time unit of the {@code tickDuration}
     * @param ticksPerWheel        the size of the wheel
     * @param leakDetection        {@code true} if leak detection should be enabled always,
     *                             if false it will only be enabled if the worker thread is not
     *                             a daemon thread.
     * @param  maxPendingTimeouts  The maximum number of pending timeouts after which call to
     *                             {@code newTimeout} will result in
     *                             {@link java.util.concurrent.RejectedExecutionException}
     *                             being thrown. No maximum pending timeouts limit is assumed if
     *                             this value is 0 or negative.
     * @throws NullPointerException     if either of {@code threadFactory} and {@code unit} is {@code null}
     * @throws IllegalArgumentException if either of {@code tickDuration} and {@code ticksPerWheel} is &lt;= 0
     */
    public HashedWheelTimer(
            ThreadFactory threadFactory,
            long tickDuration, TimeUnit unit, int ticksPerWheel, boolean leakDetection,
            long maxPendingTimeouts) {
        this(threadFactory, tickDuration, unit, ticksPerWheel, leakDetection,
                maxPendingTimeouts, ImmediateExecutor.INSTANCE);
    }
    /**
     * Creates a new timer.
     *
     * @param threadFactory        a {@link ThreadFactory} that creates a
     *                             background {@link Thread} which is dedicated to
     *                             {@link TimerTask} execution.
     * @param tickDuration         the duration between tick
     * @param unit                 the time unit of the {@code tickDuration}
     * @param ticksPerWheel        the size of the wheel
     * @param leakDetection        {@code true} if leak detection should be enabled always,
     *                             if false it will only be enabled if the worker thread is not
     *                             a daemon thread.
     * @param maxPendingTimeouts   The maximum number of pending timeouts after which call to
     *                             {@code newTimeout} will result in
     *                             {@link java.util.concurrent.RejectedExecutionException}
     *                             being thrown. No maximum pending timeouts limit is assumed if
     *                             this value is 0 or negative.
     * @param taskExecutor         The {@link Executor} that is used to execute the submitted {@link TimerTask}s.
     *                             The caller is responsible to shutdown the {@link Executor} once it is not needed
     *                             anymore.
     * @throws NullPointerException     if either of {@code threadFactory} and {@code unit} is {@code null}
     * @throws IllegalArgumentException if either of {@code tickDuration} and {@code ticksPerWheel} is &lt;= 0
     */
    public HashedWheelTimer(
            ThreadFactory threadFactory,
            long tickDuration, TimeUnit unit, int ticksPerWheel, boolean leakDetection,
            long maxPendingTimeouts, Executor taskExecutor) {

        checkNotNull(threadFactory, "threadFactory");
        checkNotNull(unit, "unit");
        checkPositive(tickDuration, "tickDuration");
        checkPositive(ticksPerWheel, "ticksPerWheel");
        this.taskExecutor = checkNotNull(taskExecutor, "taskExecutor");

        // Normalize ticksPerWheel to power of two and initialize the wheel.
        // 初始化时间轮数组，时间轮大小为大于等于 ticksPerWheel 的第一个 2 的幂，和 HashMap 类似
        wheel = createWheel(ticksPerWheel);

        // 取模用，用来定位数组中的槽
        mask = wheel.length - 1;

        // Convert tickDuration to nanos.
        // 为了保证精度，时间轮内的时间单位为纳秒
        long duration = unit.toNanos(tickDuration);

        // Prevent overflow.
        // 时间轮内的时钟拨动频率不宜太大也不宜太小
        if (duration >= Long.MAX_VALUE / wheel.length) {
            throw new IllegalArgumentException(String.format(
                    "tickDuration: %d (expected: 0 < tickDuration in nanos < %d",
                    tickDuration, Long.MAX_VALUE / wheel.length));
        }

        if (duration < MILLISECOND_NANOS) {
            logger.warn("Configured tickDuration {} smaller than {}, using 1ms.",
                        tickDuration, MILLISECOND_NANOS);
            this.tickDuration = MILLISECOND_NANOS;
        } else {
            this.tickDuration = duration;
        }

        // 创建工作线程
        workerThread = threadFactory.newThread(worker);

        // 非守护线程且 leakDetection 为 true 时检测内存是否泄漏
        leak = leakDetection || !workerThread.isDaemon() ? leakDetector.track(this) : null;

        // 初始化最大等待任务数
        this.maxPendingTimeouts = maxPendingTimeouts;

        // 如果创建的时间轮实例大于 64，打印日志，并且这个日志只会打印一次
        if (INSTANCE_COUNTER.incrementAndGet() > INSTANCE_COUNT_LIMIT &&
            WARNED_TOO_MANY_INSTANCES.compareAndSet(false, true)) {
            reportTooManyInstances();
        }
    }

    @Override
    protected void finalize() throws Throwable {
        try {
            super.finalize();
        } finally {
            // This object is going to be GCed and it is assumed the ship has sailed to do a proper shutdown. If
            // we have not yet shutdown then we want to make sure we decrement the active instance count.
            if (WORKER_STATE_UPDATER.getAndSet(this, WORKER_STATE_SHUTDOWN) != WORKER_STATE_SHUTDOWN) {
                INSTANCE_COUNTER.decrementAndGet();
            }
        }
    }

    private static HashedWheelBucket[] createWheel(int ticksPerWheel) {
        //ticksPerWheel may not be greater than 2^30
        checkInRange(ticksPerWheel, 1, 1073741824, "ticksPerWheel");

        ticksPerWheel = normalizeTicksPerWheel(ticksPerWheel);
        HashedWheelBucket[] wheel = new HashedWheelBucket[ticksPerWheel];
        for (int i = 0; i < wheel.length; i ++) {
            wheel[i] = new HashedWheelBucket();
        }
        return wheel;
    }

    private static int normalizeTicksPerWheel(int ticksPerWheel) {
        int normalizedTicksPerWheel = 1;
        while (normalizedTicksPerWheel < ticksPerWheel) {
            normalizedTicksPerWheel <<= 1;
        }
        return normalizedTicksPerWheel;
    }

    /**
     * Starts the background thread explicitly.  The background thread will
     * start automatically on demand even if you did not call this method.
     *
     * @throws IllegalStateException if this timer has been
     *                               {@linkplain #stop() stopped} already
     */
    public void start() {
        switch (WORKER_STATE_UPDATER.get(this)) {
            case WORKER_STATE_INIT:
                // 这里存在并发，通过 CAS 操作保证最终只有一个线程能开启时间轮的工作线程
                if (WORKER_STATE_UPDATER.compareAndSet(this, WORKER_STATE_INIT, WORKER_STATE_STARTED)) {
                    workerThread.start();
                }
                break;
            case WORKER_STATE_STARTED:
                break;
            case WORKER_STATE_SHUTDOWN:
                throw new IllegalStateException("cannot be started once stopped");
            default:
                throw new Error("Invalid WorkerState");
        }

        // Wait until the startTime is initialized by the worker.
        // 这里会一直阻塞，直至上面10行左右的workerThread.start()执行完成
        while (startTime == 0) {
            try {
                // startTimeInitialized 是一个 CountDownLatch，目的是为了保证工作线程的 startTime 属性初始化
                startTimeInitialized.await();
            } catch (InterruptedException ignore) {
                // Ignore - it will be ready very soon.
            }
        }
    }

    @Override
    public Set<Timeout> stop() {
        // 终止时间轮的线程不能是时间轮的工作线程
        if (Thread.currentThread() == workerThread) {
            throw new IllegalStateException(
                    HashedWheelTimer.class.getSimpleName() +
                            ".stop() cannot be called from " +
                            TimerTask.class.getSimpleName());
        }

        // 将时间轮的状态修改为 WORKER_STATE_SHUTDOWN，这里有两种情况
        // 一：时间轮是 WORKER_STATE_INIT 状态，表明时间轮从创建到终止一直没有任务进来
        // 二：时间轮是 WORKER_STATE_STARTED 状态，多个线程尝试终止时间轮，只有一个操作成功
        if (!WORKER_STATE_UPDATER.compareAndSet(this, WORKER_STATE_STARTED, WORKER_STATE_SHUTDOWN)) {
            // workerState can be 0 or 2 at this moment - let it always be 2.
            // 代码走到这里，时间轮只能是两种状态中的一个，WORKER_STATE_INIT 和 WORKER_STATE_SHUTDOWN
            // 为 WORKER_STATE_INIT 表示时间轮没有任务，因此不用返回未处理的任务，但是需要将时间轮实例 -1
            // 为 WORKER_STATE_SHUTDOWN 表示是 CAS 操作失败，什么都不用做，因为 CAS 成功的线程会处理
            if (WORKER_STATE_UPDATER.getAndSet(this, WORKER_STATE_SHUTDOWN) != WORKER_STATE_SHUTDOWN) {
                // 时间轮实例对象 -1
                INSTANCE_COUNTER.decrementAndGet();
                if (leak != null) {
                    boolean closed = leak.close(this);
                    assert closed;
                }
            }

            // CAS 操作失败，或者时间轮没有处理过任务，返回空的任务列表
            return Collections.emptySet();
        }

        try {
            boolean interrupted = false;
            while (workerThread.isAlive()) {
                // 中断时间轮工作线程
                workerThread.interrupt();
                try {
                    // 终止时间轮的线程等待时间轮工作线程 100ms，这个过程主要是为了时间轮工作线程处理未执行的任务
                    workerThread.join(100);
                } catch (InterruptedException ignored) {
                    interrupted = true;
                }
            }

            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        } finally {
            INSTANCE_COUNTER.decrementAndGet();
            if (leak != null) {
                boolean closed = leak.close(this);
                assert closed;
            }
        }
        // 返回未处理的任务
        return worker.unprocessedTimeouts();
    }

    @Override
    public Timeout newTimeout(TimerTask task, long delay, TimeUnit unit) {
        checkNotNull(task, "task");
        checkNotNull(unit, "unit");

        // 等待的任务数 +1
        long pendingTimeoutsCount = pendingTimeouts.incrementAndGet();

        // 如果时间轮内等待的任务数大于最大值，任务会被抛弃
        if (maxPendingTimeouts > 0 && pendingTimeoutsCount > maxPendingTimeouts) {
            pendingTimeouts.decrementAndGet();
            throw new RejectedExecutionException("Number of pending timeouts ("
                + pendingTimeoutsCount + ") is greater than or equal to maximum allowed pending "
                + "timeouts (" + maxPendingTimeouts + ")");
        }

        // 开启时间轮内的线程
        start();

        // Add the timeout to the timeout queue which will be processed on the next tick.
        // During processing all the queued HashedWheelTimeouts will be added to the correct HashedWheelBucket.
        // 计算当前添加任务的执行时间
        long deadline = System.nanoTime() + unit.toNanos(delay) - startTime;

        // Guard against overflow.
        if (delay > 0 && deadline < 0) {
            deadline = Long.MAX_VALUE;
        }

        // 将任务加入队列
        HashedWheelTimeout timeout = new HashedWheelTimeout(this, task, deadline);
        timeouts.add(timeout);
        return timeout;
    }

    /**
     * Returns the number of pending timeouts of this {@link Timer}.
     */
    public long pendingTimeouts() {
        return pendingTimeouts.get();
    }

    private static void reportTooManyInstances() {
        if (logger.isErrorEnabled()) {
            String resourceType = simpleClassName(HashedWheelTimer.class);
            logger.error("You are creating too many " + resourceType + " instances. " +
                    resourceType + " is a shared resource that must be reused across the JVM, " +
                    "so that only a few instances are created.");
        }
    }

    /**
     * 1、初始化startTime
     * 2、用来唤醒被阻塞的 HashedWheelTimer#start() 方法，保证 startTime 初始化
     */
    private final class Worker implements Runnable {
        private final Set<Timeout> unprocessedTimeouts = new HashSet<Timeout>();

        // 当前的tick数，每当过了tickDuration时，tick+1
        private long tick;

        @Override
        public void run() {
            // Initialize the startTime.
            // 初始化 startTime.
            startTime = System.nanoTime();
            if (startTime == 0) {
                // We use 0 as an indicator for the uninitialized value here, so make sure it's not 0 when initialized.
                startTime = 1;
            }

            // Notify the other threads waiting for the initialization at start().
            // 用来唤醒被阻塞的 HashedWheelTimer#start() 方法，保证 startTime 初始化
            startTimeInitialized.countDown();

            // 死循环执行，直到时间轮被关闭
            do {
                // 时钟拨动
                // waitForNextTick()返回的是currentTime，（距离程序启动已经过去多久了）
                final long deadline = waitForNextTick();
                if (deadline > 0) {
                    int idx = (int) (tick & mask);
                    // 移除被取消的任务
                    processCancelledTasks();
                    HashedWheelBucket bucket =
                            wheel[idx];
                    // 将任务加载进时间轮
                    transferTimeoutsToBuckets();
                    // 执行当前时间轮槽内的任务
                    bucket.expireTimeouts(deadline);
                    tick++;
                }
            } while (WORKER_STATE_UPDATER.get(HashedWheelTimer.this) == WORKER_STATE_STARTED);

            // Fill the unprocessedTimeouts so we can return them from stop() method.
            // 时间轮关闭，将还未执行的任务以列表的形式保存到 unprocessedTimeouts 集合中，在 stop 方法中返回出去
            // 还未执行的任务可能会在两个地方，一：时间轮数组内，二：队列中
            for (HashedWheelBucket bucket: wheel) {
                bucket.clearTimeouts(unprocessedTimeouts);
            }
            for (;;) {
                HashedWheelTimeout timeout = timeouts.poll();
                if (timeout == null) {
                    break;
                }
                if (!timeout.isCancelled()) {
                    unprocessedTimeouts.add(timeout);
                }
            }
            // 处理过期的任务
            processCancelledTasks();
        }

        private void transferTimeoutsToBuckets() {
            // transfer only max. 100000 timeouts per tick to prevent a thread to stale the workerThread when it just
            // adds new timeouts in a loop.
            // 一次最多只处理队列中的 100000 个任务
            for (int i = 0; i < 100000; i++) {
                HashedWheelTimeout timeout = timeouts.poll();
                if (timeout == null) {
                    // all processed
                    break;
                }
                // 过滤已经取消的任务
                if (timeout.state() == HashedWheelTimeout.ST_CANCELLED) {
                    // Was cancelled in the meantime.
                    continue;
                }

                /**
                 * 下面这段在干什么，就是先计算这个任务的时间，并且需要过几轮，到轮中的第几个（有点类似于除了后，还要取余）
                 */
                // 计算当前任务到执行还需要经过几次时钟拨动
                // 假设时间轮数组大小是 10，calculated 为 12，需要时间轮转动一圈加两次时钟拨动后后才能执行这个任务，因此还需要计算一下圈数
                long calculated = timeout.deadline / tickDuration;
                // 计算当前任务到执行还需要经过几圈时钟拨动
                timeout.remainingRounds = (calculated - tick) / wheel.length;
                // 有的任务可能在队列里很长时间，时间过期了也没有被调度，将这种情况的任务放在当前轮次内执行
                final long ticks = Math.max(calculated, tick); // Ensure we don't schedule for past.
                // 计算任务在时间轮数组中的槽
                int stopIndex = (int) (ticks & mask);

                HashedWheelBucket bucket = wheel[stopIndex];
                // 将任务放到时间轮的数组中，多个任务可能定位时间轮的同一个槽，这些任务通过以链表的形式链接
                bucket.addTimeout(timeout);
            }
        }

        private void processCancelledTasks() {
            for (;;) {
                HashedWheelTimeout timeout = cancelledTimeouts.poll();
                if (timeout == null) {
                    // all processed
                    break;
                }
                try {
                    timeout.remove();
                } catch (Throwable t) {
                    if (logger.isWarnEnabled()) {
                        logger.warn("An exception was thrown while process a cancellation task", t);
                    }
                }
            }
        }

        /**
         * calculate goal nanoTime from startTime and current tick number,
         * then wait until that goal has been reached.
         * @return Long.MIN_VALUE if received a shutdown request,
         * current time otherwise (with Long.MIN_VALUE changed by +1)
         * 时间对齐，即对齐为下一个tickDuration
         */
        private long waitForNextTick() {
            // 计算时钟下次拨动的相对时间
            long deadline = tickDuration * (tick + 1);

            for (;;) {
                // 获取当前时间的相对时间（距离程序启动已经过去多久了）
                final long currentTime = System.nanoTime() - startTime;
                // 计算距离时钟下次拨动的时间
                // 这里之所以加 999999 后再除 10000000, 是为了保证足够的 sleep 时间
                // 例如：当 deadline - currentTime = 2000002 的时候，如果不加 999999，则只睡了 2ms
                // 而 2ms 其实是未到达 deadline 时间点的，所以为了使上述情况能 sleep 足够的时间，加上 999999 后，会多睡 1ms
                // 我理解就是向上取整
                long sleepTimeMs = (deadline - currentTime + 999999) / 1000000;

                // <=0 说明可以拨动时钟了
                if (sleepTimeMs <= 0) {
                    if (currentTime == Long.MIN_VALUE) {
                        return -Long.MAX_VALUE;
                    } else {
                        return currentTime;
                    }
                }

                // Check if we run on windows, as if thats the case we will need
                // to round the sleepTime as workaround for a bug that only affect
                // the JVM if it runs on windows.
                //
                // See https://github.com/netty/netty/issues/356
                // 这里是为了兼容 Windows 平台，因为 Windows 平台的调度最小单位为 10ms，如果不是 10ms 的倍数，可能会引起 sleep 时间不准确
                if (PlatformDependent.isWindows()) {
                    sleepTimeMs = sleepTimeMs / 10 * 10;
                    if (sleepTimeMs == 0) {
                        sleepTimeMs = 1;
                    }
                }

                try {
                    // sleep 到下次时钟拨动
                    Thread.sleep(sleepTimeMs);
                } catch (InterruptedException ignored) {
                    if (WORKER_STATE_UPDATER.get(HashedWheelTimer.this) == WORKER_STATE_SHUTDOWN) {
                        return Long.MIN_VALUE;
                    }
                }
            }
        }

        public Set<Timeout> unprocessedTimeouts() {
            return Collections.unmodifiableSet(unprocessedTimeouts);
        }
    }

    private static final class HashedWheelTimeout implements Timeout, Runnable {

        private static final int ST_INIT = 0;
        private static final int ST_CANCELLED = 1;
        private static final int ST_EXPIRED = 2;
        private static final AtomicIntegerFieldUpdater<HashedWheelTimeout> STATE_UPDATER =
                AtomicIntegerFieldUpdater.newUpdater(HashedWheelTimeout.class, "state");

        private final HashedWheelTimer timer;
        private final TimerTask task;
        private final long deadline;

        @SuppressWarnings({"unused", "FieldMayBeFinal", "RedundantFieldInitialization" })
        private volatile int state = ST_INIT;

        // remainingRounds will be calculated and set by Worker.transferTimeoutsToBuckets() before the
        // HashedWheelTimeout will be added to the correct HashedWheelBucket.
        long remainingRounds;

        // This will be used to chain timeouts in HashedWheelTimerBucket via a double-linked-list.
        // As only the workerThread will act on it there is no need for synchronization / volatile.
        HashedWheelTimeout next;
        HashedWheelTimeout prev;

        // The bucket to which the timeout was added
        HashedWheelBucket bucket;

        HashedWheelTimeout(HashedWheelTimer timer, TimerTask task, long deadline) {
            this.timer = timer;
            this.task = task;
            this.deadline = deadline;
        }

        @Override
        public Timer timer() {
            return timer;
        }

        @Override
        public TimerTask task() {
            return task;
        }

        @Override
        public boolean cancel() {
            // only update the state it will be removed from HashedWheelBucket on next tick.
            if (!compareAndSetState(ST_INIT, ST_CANCELLED)) {
                return false;
            }
            // If a task should be canceled we put this to another queue which will be processed on each tick.
            // So this means that we will have a GC latency of max. 1 tick duration which is good enough. This way
            // we can make again use of our MpscLinkedQueue and so minimize the locking / overhead as much as possible.
            timer.cancelledTimeouts.add(this);
            return true;
        }

        void remove() {
            HashedWheelBucket bucket = this.bucket;
            if (bucket != null) {
                bucket.remove(this);
            } else {
                timer.pendingTimeouts.decrementAndGet();
            }
        }

        public boolean compareAndSetState(int expected, int state) {
            return STATE_UPDATER.compareAndSet(this, expected, state);
        }

        public int state() {
            return state;
        }

        @Override
        public boolean isCancelled() {
            return state() == ST_CANCELLED;
        }

        @Override
        public boolean isExpired() {
            return state() == ST_EXPIRED;
        }

        public void expire() {
            // 任务状态校验
            if (!compareAndSetState(ST_INIT, ST_EXPIRED)) {
                return;
            }

            try {
                timer.taskExecutor.execute(this);
            } catch (Throwable t) {
                if (logger.isWarnEnabled()) {
                    logger.warn("An exception was thrown while submit " + TimerTask.class.getSimpleName()
                            + " for execution.", t);
                }
            }
        }

        @Override
        public void run() {
            try {
                task.run(this);
            } catch (Throwable t) {
                if (logger.isWarnEnabled()) {
                    logger.warn("An exception was thrown by " + TimerTask.class.getSimpleName() + '.', t);
                }
            }
        }

        @Override
        public String toString() {
            final long currentTime = System.nanoTime();
            long remaining = deadline - currentTime + timer.startTime;

            StringBuilder buf = new StringBuilder(192)
               .append(simpleClassName(this))
               .append('(')
               .append("deadline: ");
            if (remaining > 0) {
                buf.append(remaining)
                   .append(" ns later");
            } else if (remaining < 0) {
                buf.append(-remaining)
                   .append(" ns ago");
            } else {
                buf.append("now");
            }

            if (isCancelled()) {
                buf.append(", cancelled");
            }

            return buf.append(", task: ")
                      .append(task())
                      .append(')')
                      .toString();
        }
    }

    /**
     * Bucket that stores HashedWheelTimeouts. These are stored in a linked-list like datastructure to allow easy
     * removal of HashedWheelTimeouts in the middle. Also the HashedWheelTimeout act as nodes themself and so no
     * extra object creation is needed.
     */
    private static final class HashedWheelBucket {
        // Used for the linked-list datastructure
        private HashedWheelTimeout head;
        private HashedWheelTimeout tail;

        /**
         * Add {@link HashedWheelTimeout} to this bucket.
         */
        public void addTimeout(HashedWheelTimeout timeout) {
            assert timeout.bucket == null;
            // 任务构成双向链表
            timeout.bucket = this;
            if (head == null) {
                head = tail = timeout;
            } else {
                tail.next = timeout;
                timeout.prev = tail;
                tail = timeout;
            }
        }

        /**
         * Expire all {@link HashedWheelTimeout}s for the given {@code deadline}.
         * 如果圈数小于等于0，表明就是这一圈该执行的，那就执行吧
         * 如果圈数大于0，那就表明这一个不该执行，该下一圈执行，那就直接减圈数即可
         */
        public void expireTimeouts(long deadline) {
            HashedWheelTimeout timeout = head;

            // process all timeouts
            while (timeout != null) {
                HashedWheelTimeout next = timeout.next;
                // 任务执行的圈数 > 0，表示任务还需要经过 remainingRounds 圈时钟循环才能执行
                if (timeout.remainingRounds <= 0) {
                    // 从链表中移除当前任务，并返回链表中下一个任务
                    next = remove(timeout);
                    if (timeout.deadline <= deadline) {
                        // 执行任务
                        timeout.expire();
                    } else {
                        // The timeout was placed into a wrong slot. This should never happen.
                        throw new IllegalStateException(String.format(
                                "timeout.deadline (%d) > deadline (%d)", timeout.deadline, deadline));
                    }
                } else if (timeout.isCancelled()) {
                    // 过滤取消的任务
                    next = remove(timeout);
                } else {
                    // 圈数 -1
                    timeout.remainingRounds --;
                }
                timeout = next;
            }
        }

        public HashedWheelTimeout remove(HashedWheelTimeout timeout) {
            HashedWheelTimeout next = timeout.next;
            // remove timeout that was either processed or cancelled by updating the linked-list
            if (timeout.prev != null) {
                timeout.prev.next = next;
            }
            if (timeout.next != null) {
                timeout.next.prev = timeout.prev;
            }

            if (timeout == head) {
                // if timeout is also the tail we need to adjust the entry too
                if (timeout == tail) {
                    tail = null;
                    head = null;
                } else {
                    head = next;
                }
            } else if (timeout == tail) {
                // if the timeout is the tail modify the tail to be the prev node.
                tail = timeout.prev;
            }
            // null out prev, next and bucket to allow for GC.
            timeout.prev = null;
            timeout.next = null;
            timeout.bucket = null;
            timeout.timer.pendingTimeouts.decrementAndGet();
            return next;
        }

        /**
         * Clear this bucket and return all not expired / cancelled {@link Timeout}s.
         */
        public void clearTimeouts(Set<Timeout> set) {
            for (;;) {
                HashedWheelTimeout timeout = pollTimeout();
                if (timeout == null) {
                    return;
                }
                if (timeout.isExpired() || timeout.isCancelled()) {
                    continue;
                }
                set.add(timeout);
            }
        }

        private HashedWheelTimeout pollTimeout() {
            HashedWheelTimeout head = this.head;
            if (head == null) {
                return null;
            }
            HashedWheelTimeout next = head.next;
            if (next == null) {
                tail = this.head =  null;
            } else {
                this.head = next;
                next.prev = null;
            }

            // null out prev and next to allow for GC.
            head.next = null;
            head.prev = null;
            head.bucket = null;
            return head;
        }
    }
}
