import sys
import threading
import time
import numpy as np
import pygame

class HeadlessTeleop:
    """Headless teleop with optional pygame joystick support."""

    def __init__(
        self,
        config_init=(0.0, 0.0, 0.0),
        lin_step=0.2,
        ang_step=0.2,
        max_lin=1.0,
        max_ang=3.0,
        height_init=0.3,
        height_step=0.01,
        min_height=0.2,
        max_height=0.35,
        gamepad_deadzone=0.1,
    ):
        self.cmd_vel = np.array(config_init, dtype=np.float32)
        self.cmd_height = float(height_init)

        self.lin_step = float(lin_step)
        self.ang_step = float(ang_step)
        self.height_step = float(height_step)

        self.max_lin = float(max_lin)
        self.max_ang = float(max_ang)
        self.min_height = float(min_height)
        self.max_height = float(max_height)
        self.gamepad_deadzone = float(gamepad_deadzone)

        self.lock = threading.Lock()
        self.running = True

        # Initialize Pygame Joystick subsystem
        pygame.init()
        pygame.joystick.init()

        self._gamepad_thread = None
        self._start_gamepad_thread_if_available()

        print("Headless teleop active.")
        if self._gamepad_thread is not None:
            print("Gamepad: Left Stick (Y) -> Linear, Right Stick (X) -> Yaw, D-pad -> Height, Button 0 -> Stop.")
        else:
            print("No gamepad detected.")

    def get_command(self):
        with self.lock:
            return self.cmd_vel.copy()

    def get_height_command(self):
        with self.lock:
            return self.cmd_height

    def close(self):
        self.running = False
        if self._gamepad_thread is not None:
            self._gamepad_thread.join(timeout=0.2)
        pygame.joystick.quit()
        pygame.quit()

    def _start_gamepad_thread_if_available(self):
        if pygame.joystick.get_count() > 0:
            self._gamepad_thread = threading.Thread(target=self._gamepad_loop, daemon=True)
            self._gamepad_thread.start()

    def _apply_deadzone(self, value):
        if abs(value) < self.gamepad_deadzone:
            return 0.0
        return value

    def _gamepad_loop(self):
        # Initialize the first available joystick
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        
        # Clock to throttle the loop (prevents 100% CPU usage)
        clock = pygame.time.Clock()

        while self.running:
            # Pygame requires pumping the event queue to update joystick states
            pygame.event.pump()

            with self.lock:
                # 1. Linear Velocity (Left Stick Y - Axis 1)
                # Note: Pygame sticks are -1.0 (Up) to 1.0 (Down), so we negate it
                raw_ly = -joystick.get_axis(1)
                self.cmd_vel[0] = self._apply_deadzone(raw_ly) * self.max_lin

                # 2. Yaw (Right Stick X - Axis 3 or 2 depending on controller)
                # We try axis 3 first (Xbox standard), fallback to 2
                axis_x = joystick.get_axis(3) if joystick.get_numaxes() > 3 else joystick.get_axis(2)
                self.cmd_vel[2] = -self._apply_deadzone(axis_x) * self.max_ang

                # 3. Height Control (D-Pad / Hat 0)
                # get_hat returns (x, y). y=1 is Up, y=-1 is Down.
                if joystick.get_numhats() > 0:
                    hat = joystick.get_hat(0)
                    if hat[1] == 1: # Up
                        self.cmd_height = np.clip(self.cmd_height + self.height_step, self.min_height, self.max_height)
                    elif hat[1] == -1: # Down
                        self.cmd_height = np.clip(self.cmd_height - self.height_step, self.min_height, self.max_height)

                # 4. Emergency Stop (Button 0 - usually 'A' or 'Cross')
                if joystick.get_button(0):
                    self.cmd_vel[:] = 0.0

            clock.tick(50)  # Run at 50Hz

# Example usage:
if __name__ == "__main__":
    teleop = HeadlessTeleop()
    try:
        while True:
            print(f"Vel: {teleop.get_command()} | Height: {teleop.get_height_command():.2f}", end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        teleop.close()