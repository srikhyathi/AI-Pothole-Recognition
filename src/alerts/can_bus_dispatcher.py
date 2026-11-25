"""
Autonomous Vehicle Safety Alert & CAN Bus J1939 Telemetry Dispatcher.
Calculates dynamic Time-to-Collision (TTC) based on vehicle speedometer,
formats standard CAN bus J1939 priority frames, and modulates audio warning pulses.
"""

from typing import Dict, Any, Optional
import math

class CANBusAlertDispatcher:
    """Dispatches vehicle safety alerts across vehicle ECU CAN network."""

    def __init__(self, vehicle_speed_kmh: float = 60.0):
        self.vehicle_speed_kmh = vehicle_speed_kmh

    @property
    def speed_mps(self) -> float:
        """Vehicle velocity in meters per second."""
        return (self.vehicle_speed_kmh * 1000.0) / 3600.0

    def compute_time_to_collision(self, distance_ahead_m: float) -> float:
        """
        Compute Time-to-Collision: TTC = Distance / Velocity.
        Returns time in seconds.
        """
        if self.speed_mps <= 0.5:
            return 99.0
        return max(0.1, distance_ahead_m / self.speed_mps)

    def calculate_braking_safety_margin(self, distance_ahead_m: float, deceleration_mps2: float = 4.5) -> Dict[str, Any]:
        """
        Calculate required stopping distance vs current distance.
        Stopping Distance d_stop = (v^2) / (2 * a) + (v * reaction_time)
        """
        reaction_time_s = 0.8
        v = self.speed_mps
        stopping_distance_m = ((v * v) / (2.0 * deceleration_mps2)) + (v * reaction_time_s)
        margin_m = distance_ahead_m - stopping_distance_m
        ttc = self.compute_time_to_collision(distance_ahead_m)

        if margin_m < 0:
            urgency = "IMMINENT_CRITICAL"
            action = "EMERGENCY_EVASION_OR_HARD_BRAKE"
            buzzer_frequency_hz = 2500
        elif ttc < 2.5:
            urgency = "HIGH"
            action = "MODERATE_BRAKE_RECOMMENDED"
            buzzer_frequency_hz = 1500
        else:
            urgency = "ADVISORY"
            action = "PREPARE_CAUTION"
            buzzer_frequency_hz = 800

        return {
            "distance_ahead_m": round(distance_ahead_m, 2),
            "vehicle_speed_kmh": round(self.vehicle_speed_kmh, 1),
            "time_to_collision_s": round(ttc, 2),
            "stopping_distance_m": round(stopping_distance_m, 2),
            "safety_margin_m": round(margin_m, 2),
            "urgency": urgency,
            "recommended_action": action,
            "buzzer_frequency_hz": buzzer_frequency_hz
        }

    def format_j1939_can_frame(self, pgn: int = 0xFEF1, severity_code: int = 3, distance_m: float = 15.0) -> Dict[str, Any]:
        """
        Format standard SAE J1939 29-bit identifier and 8-byte payload frame.
        Priority = 3 (High Priority Hazard Message).
        """
        # Encode distance in units of 0.1m
        dist_raw = min(65535, int(distance_m * 10))
        dist_byte_low = dist_raw & 0xFF
        dist_byte_high = (dist_raw >> 8) & 0xFF

        payload = [
            0xAA,                 # Header: Hazard Detected flag
            severity_code & 0xFF, # Byte 1: Severity (1=Low, 2=Med, 3=Crit)
            dist_byte_low,        # Byte 2: Distance Low
            dist_byte_high,       # Byte 3: Distance High
            int(self.vehicle_speed_kmh) & 0xFF, # Byte 4: Speed
            0x00,                 # Byte 5: Reserved
            0x00,                 # Byte 6: Reserved
            0xFF                  # Byte 7: Checksum/Termination
        ]

        can_id = 0x18000000 | ((pgn & 0xFFFF) << 8) | 0x01
        hex_payload = " ".join(f"{b:02X}" for b in payload)

        return {
            "can_id_hex": f"0x{can_id:08X}",
            "pgn": hex(pgn),
            "payload_bytes": payload,
            "payload_hex": hex_payload,
            "protocol": "SAE_J1939_CAN_2.0B"
        }
