from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dqn_agent import _action_name, _action_to_int, _artifacts, _load_qrdqn, _select_model_path, resolve_torch_device
from .observations import build_dino_observation

REAL_GAME_STATE_SCRIPT = r"""
const RunnerCtor = (typeof Runner !== 'undefined') ? Runner : (window.Runner || null);
if (typeof loadTimeData !== 'undefined' && typeof loadTimeDataRaw !== 'undefined' && !loadTimeData.data_) {
  try {
    loadTimeData.data = loadTimeDataRaw;
  } catch (e) {
    // Ignore if Chrome already hydrated it.
  }
}
if (RunnerCtor && !document.querySelector('.runner-container') && document.querySelector('.interstitial-wrapper')) {
  try {
    if (typeof RunnerCtor.getInstance === 'function') {
      RunnerCtor.initializeInstance('.interstitial-wrapper');
    } else if (typeof RunnerCtor.initialize === 'function') {
      RunnerCtor.initialize('.interstitial-wrapper');
    }
  } catch (e) {
    // Ignore bootstrap errors; state read below will fail if runner is still unavailable.
  }
}
let runner = null;
if (RunnerCtor && typeof RunnerCtor.getInstance === 'function') {
  try {
    runner = RunnerCtor.getInstance();
  } catch (e) {
    runner = null;
  }
}
runner =
  runner ||
  (RunnerCtor && (RunnerCtor.instance_ || RunnerCtor.instance)) ||
  null;
if (!runner) {
  return null;
}
const tRex = runner.tRex || {};
const dims = runner.dimensions || {};
const obstacles = (runner.horizon && runner.horizon.obstacles) ? runner.horizon.obstacles : [];
return {
  activated: !!runner.activated,
  crashed: !!runner.crashed,
  playing: !!runner.playing,
  currentSpeed: Number(runner.currentSpeed || 0),
  distanceRan: Number(runner.distanceRan || 0),
  dimensions: {
    width: Number(dims.WIDTH || dims.width || 600),
    height: Number(dims.HEIGHT || dims.height || 150),
  },
  tRex: {
    xPos: Number(tRex.xPos || 0),
    yPos: Number(tRex.yPos || 0),
    groundYPos: Number(tRex.groundYPos || 0),
    ducking: !!tRex.ducking,
    jumping: !!tRex.jumping,
    speedDrop: !!tRex.speedDrop,
  },
  obstacles: obstacles.map((obstacle) => ({
    xPos: Number(obstacle.xPos || 0),
    yPos: Number(obstacle.yPos || 0),
    width: Number(obstacle.width || (obstacle.typeConfig && obstacle.typeConfig.width) || 0),
    height: Number((obstacle.typeConfig && obstacle.typeConfig.height) || obstacle.height || 0),
    type: String(obstacle.type || ''),
  })),
};
"""


@dataclass(slots=True)
class RealGameRunResult:
    model_path: Path
    steps: int
    episodes: int
    crashes: int
    passed_obstacles: int
    action_counts: dict[str, int]
    url: str


@dataclass(slots=True)
class _LiveObsState:
    player_y_px: float
    floor_y_px: float
    player_vy_px_s: float
    grounded: bool
    ducking: bool
    current_speed_px_s: float
    screen_width_px: float
    screen_height_px: float
    nearest_distance_px: float | None
    nearest_width_px: float | None
    nearest_height_px: float | None
    nearest_is_bird: bool
    second_distance_px: float | None
    second_width_px: float | None
    second_height_px: float | None
    obstacle_count: int
    passed_obstacles: int
    crashed: bool


def play_real_dqn(
    *,
    model_path: str | Path | None = None,
    best: bool = False,
    duration: float = 30.0,
    output_dir: str | Path | None = None,
    device: str = "auto",
    url: str = "chrome://dino/",
    chrome_binary: str | None = None,
    headless: bool = False,
    window_width: int = 1000,
    window_height: int = 320,
    restart_on_crash: bool = True,
) -> RealGameRunResult:
    artifacts = _artifacts(output_dir)
    selected_model_path = _select_model_path(artifacts, model_path, best=best)
    QRDQN = _load_qrdqn()
    driver = _launch_chrome(
        url=url,
        chrome_binary=chrome_binary,
        headless=headless,
        window_width=window_width,
        window_height=window_height,
    )
    model = QRDQN.load(str(selected_model_path), device=resolve_torch_device(device))

    action_counts: Counter[str] = Counter()
    steps = 0
    episodes = 1
    crashes = 0
    passed_obstacles = 0
    down_pressed = False
    prev_player_y: float | None = None
    prev_time: float | None = None

    try:
        _focus_and_start_game(driver)
        deadline = time.time() + max(0.0, float(duration))
        while time.time() < deadline:
            raw_state = driver.execute_script(REAL_GAME_STATE_SCRIPT)
            if raw_state is None:
                raise RuntimeError(
                    "Could not read Dino runtime state from the page. "
                    "If `chrome://dino` is blocked by your browser setup, try `--url` with a compatible Dino page."
                )

            live_state = _map_live_state(
                raw_state=raw_state,
                prev_player_y=prev_player_y,
                prev_time=prev_time,
            )
            prev_player_y = live_state.player_y_px
            prev_time = time.time()
            passed_obstacles = max(passed_obstacles, live_state.passed_obstacles)

            if live_state.crashed:
                crashes += 1
                if restart_on_crash:
                    _restart_game(driver)
                    episodes += 1
                    down_pressed = False
                    prev_player_y = None
                    prev_time = None
                    time.sleep(0.2)
                    continue
                break

            observation = build_dino_observation(
                player_y_px=live_state.player_y_px,
                player_vy_px=live_state.player_vy_px_s,
                grounded=live_state.grounded,
                ducking=live_state.ducking,
                current_speed_px_s=live_state.current_speed_px_s,
                screen_width_px=live_state.screen_width_px,
                screen_height_px=live_state.screen_height_px,
                floor_y_px=live_state.floor_y_px,
                max_speed_px_s=780.0,
                nearest_distance_px=live_state.nearest_distance_px,
                nearest_width_px=live_state.nearest_width_px,
                nearest_height_px=live_state.nearest_height_px,
                nearest_is_bird=live_state.nearest_is_bird,
                second_distance_px=live_state.second_distance_px,
                second_width_px=live_state.second_width_px,
                second_height_px=live_state.second_height_px,
                obstacle_count=live_state.obstacle_count,
            )
            action, _ = model.predict(observation, deterministic=True)
            action_int = _action_to_int(action)
            action_counts[_action_name(action_int)] += 1
            down_pressed = _apply_real_action(driver, action_int, down_pressed)
            steps += 1
            time.sleep(1.0 / 30.0)
    finally:
        try:
            if down_pressed:
                _release_duck(driver)
        finally:
            driver.quit()

    return RealGameRunResult(
        model_path=selected_model_path,
        steps=steps,
        episodes=episodes,
        crashes=crashes,
        passed_obstacles=passed_obstacles,
        action_counts=dict(action_counts),
        url=url,
    )


def _map_live_state(
    *,
    raw_state: dict[str, Any],
    prev_player_y: float | None,
    prev_time: float | None,
) -> _LiveObsState:
    dims = raw_state.get("dimensions") or {}
    t_rex = raw_state.get("tRex") or {}
    obstacles = list(raw_state.get("obstacles") or [])

    screen_width = float(dims.get("width") or 600.0)
    screen_height = float(dims.get("height") or 150.0)
    floor_y = screen_height - 10.0
    stand_height = 47.0

    ground_y = float(t_rex.get("groundYPos") or 0.0)
    y_pos = float(t_rex.get("yPos") or ground_y)
    airborne_height = max(0.0, ground_y - y_pos)
    player_y = floor_y - stand_height - airborne_height

    now = time.time()
    if prev_player_y is None or prev_time is None:
        player_vy = 0.0
    else:
        dt = max(1e-6, now - prev_time)
        player_vy = (player_y - prev_player_y) / dt

    speed_units = float(raw_state.get("currentSpeed") or 0.0)
    current_speed_px_s = speed_units * 60.0
    player_x = float(t_rex.get("xPos") or 50.0)
    player_width = 59.0 if bool(t_rex.get("ducking")) else 44.0

    ahead = []
    for obstacle in obstacles:
        obstacle_x = float(obstacle.get("xPos") or 0.0)
        obstacle_width = float(obstacle.get("width") or 0.0)
        if obstacle_x + obstacle_width >= player_x - 20.0:
            ahead.append(obstacle)
    ahead.sort(key=lambda obstacle: float(obstacle.get("xPos") or 0.0))

    def _distance(obstacle: dict[str, Any] | None) -> float | None:
        if obstacle is None:
            return None
        return max(0.0, float(obstacle.get("xPos") or 0.0) - player_x - player_width)

    nearest = ahead[0] if ahead else None
    second = ahead[1] if len(ahead) > 1 else None
    passed_estimate = int(float(raw_state.get("distanceRan") or 0.0) / 120.0)

    return _LiveObsState(
        player_y_px=player_y,
        floor_y_px=floor_y,
        player_vy_px_s=player_vy,
        grounded=not bool(t_rex.get("jumping")),
        ducking=bool(t_rex.get("ducking")),
        current_speed_px_s=current_speed_px_s,
        screen_width_px=screen_width,
        screen_height_px=screen_height,
        nearest_distance_px=_distance(nearest),
        nearest_width_px=float(nearest.get("width")) if nearest is not None else None,
        nearest_height_px=float(nearest.get("height")) if nearest is not None else None,
        nearest_is_bird=(str(nearest.get("type") or "").upper() == "PTERODACTYL") if nearest is not None else False,
        second_distance_px=_distance(second),
        second_width_px=float(second.get("width")) if second is not None else None,
        second_height_px=float(second.get("height")) if second is not None else None,
        obstacle_count=len(ahead),
        passed_obstacles=passed_estimate,
        crashed=bool(raw_state.get("crashed")),
    )


def _launch_chrome(
    *,
    url: str,
    chrome_binary: str | None,
    headless: bool,
    window_width: int,
    window_height: int,
):
    try:
        from selenium import webdriver
        from selenium.common.exceptions import WebDriverException
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
    except Exception as exc:
        raise RuntimeError(
            "Selenium is required for real-game play. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    driver = _build_driver(
        webdriver=webdriver,
        chrome_binary=chrome_binary,
        headless=headless,
        window_width=window_width,
        window_height=window_height,
    )
    try:
        driver.get(url)
    except WebDriverException as exc:
        message = str(exc)
        if not (
            url.startswith("chrome://dino")
            and (
                "ERR_INTERNET_DISCONNECTED" in message
                or "ERR_NAME_NOT_RESOLVED" in message
                or "ERR_CONNECTION" in message
            )
        ):
            raise RuntimeError(
                "Could not open the official Dino page at `chrome://dino/` under Selenium in this Chrome setup."
            ) from exc

    try:
        WebDriverWait(driver, 10).until(
            lambda d: d.execute_script(
                "return !!((typeof Runner !== 'undefined') || (window.Runner));"
            )
        )
    except Exception:
        body = driver.find_element(By.TAG_NAME, "body")
        body.click()
        time.sleep(1.0)

    return driver


def _build_driver(
    *,
    webdriver: Any,
    chrome_binary: str | None,
    headless: bool,
    window_width: int,
    window_height: int,
):
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-background-timer-throttling")
    options.add_argument("--disable-renderer-backgrounding")
    options.add_argument("--disable-backgrounding-occluded-windows")
    options.add_argument("--autoplay-policy=no-user-gesture-required")
    if headless:
        options.add_argument("--headless=new")
    if chrome_binary:
        options.binary_location = chrome_binary

    driver = webdriver.Chrome(options=options)
    driver.set_window_size(int(window_width), int(window_height))
    return driver


def _focus_and_start_game(driver: Any) -> None:
    body = driver.find_element("tag name", "body")
    body.click()
    time.sleep(0.2)
    _tap_key(driver, "Space", 32)
    time.sleep(0.2)


def _restart_game(driver: Any) -> None:
    _tap_key(driver, "Space", 32)


def _release_duck(driver: Any) -> None:
    _dispatch_key(driver, "keyup", "ArrowDown", 40)


def _apply_real_action(driver: Any, action: int, down_pressed: bool) -> bool:
    if action == 1:
        if down_pressed:
            _dispatch_key(driver, "keyup", "ArrowDown", 40)
            down_pressed = False
        _tap_key(driver, "ArrowUp", 38)
        return down_pressed

    if action == 2:
        if not down_pressed:
            _dispatch_key(driver, "keydown", "ArrowDown", 40)
        return True

    if down_pressed:
        _dispatch_key(driver, "keyup", "ArrowDown", 40)
    return False


def _tap_key(driver: Any, key: str, key_code: int) -> None:
    _dispatch_key(driver, "keydown", key, key_code)
    _dispatch_key(driver, "keyup", key, key_code)


def _dispatch_key(driver: Any, event_type: str, key: str, key_code: int) -> None:
    driver.execute_script(
        """
        const [eventType, key, keyCode] = arguments;
        const target = document.body || document.documentElement;
        const event = new KeyboardEvent(eventType, {
          key,
          code: key,
          keyCode,
          which: keyCode,
          bubbles: true,
        });
        Object.defineProperty(event, 'keyCode', {get: () => keyCode});
        Object.defineProperty(event, 'which', {get: () => keyCode});
        target.dispatchEvent(event);
        """,
        event_type,
        key,
        int(key_code),
    )
