import pygame

class GamepadHandler:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        self.joystick_count = pygame.joystick.get_count()
        self.joysticks = []

        for i in range(self.joystick_count):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            self.joysticks.append(joystick)
            print(f"Initialized Joystick {i}: {joystick.get_name()}")

    def get_joystick_count(self):
        return self.joystick_count

    def get_joystick_info(self, index):
        if index < self.joystick_count:
            joystick = self.joysticks[index]
            info = {
                "name": joystick.get_name(),
                "axes": joystick.get_numaxes(),
                "buttons": joystick.get_numbuttons(),
                "hats": joystick.get_numhats()
            }
            return info
        else:
            raise IndexError("Joystick index out of range")

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.JOYAXISMOTION:
                print(f"Axis motion: {event.axis}, Value: {event.value}")
            elif event.type == pygame.JOYBUTTONDOWN:
                print(f"Button pressed: {event.button}")
            elif event.type == pygame.JOYBUTTONUP:
                print(f"Button released: {event.button}")
            elif event.type == pygame.JOYHATMOTION:
                print(f"Hat motion: {event.hat}, Value: {event.value}")
        return True

    def quit(self):
        pygame.quit()

# 使用示例
def main():
    handler = GamepadHandler()
    if handler.get_joystick_count() == 0:
        print("No joysticks connected.")
    else:
        for i in range(handler.get_joystick_count()):
            info = handler.get_joystick_info(i)
            print(f"Joystick {i}: {info}")

    running = True
    while running:
        running = handler.process_events()

    handler.quit()

if __name__ == "__main__":
    main()
