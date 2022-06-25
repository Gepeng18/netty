package io.netty.util.gepeng;

import io.netty.util.HashedWheelTimer;
import io.netty.util.Timeout;
import io.netty.util.TimerTask;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class WheelTimerTest {

	private static final HashedWheelTimerInstance INSTANCE = HashedWheelTimerInstance.INSTANCE;

	public static void main(String[] args) throws IOException {

		INSTANCE.getWheelTimer().newTimeout(new PrintTimerTask(), 3, TimeUnit.SECONDS);
		System.in.read();
	}


	static class PrintTimerTask implements TimerTask {
		@Override
		public void run(Timeout timeout) {
			System.out.println("Hello world");
		}
	}

	enum HashedWheelTimerInstance {
		INSTANCE;
		private final HashedWheelTimer wheelTimer;

		HashedWheelTimerInstance() {
			wheelTimer = new HashedWheelTimer(r -> {
				Thread t = new Thread(r);
				t.setUncaughtExceptionHandler((t1, e) -> System.out.println(t1.getName() + e.getMessage()));
				t.setName("-HashedTimerWheelInstance-");
				return t;
			}, 100, TimeUnit.MILLISECONDS, 64);
		}

		public HashedWheelTimer getWheelTimer() {
			return wheelTimer;
		}
	}
}
