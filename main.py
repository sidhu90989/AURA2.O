"""AURA Orchestrator

Refactored lightweight orchestrator focused on:
 - Safe startup under constrained environments (disk, network)
 - Environment flag driven behavior (HEADLESS, MINIMAL_MODE, RUN_DURATION, HEADLESS_COMMANDS)
 - Dependency injection with graceful degradation (stubs when heavy deps/services unavailable)
 - Simple command routing and memory storage
 - Heartbeat logging & structured shutdown

Heavy cognitive/planning layers can be added later; this file establishes
the execution spine needed for iterative enhancement.
"""

from __future__ import annotations

import os
import time
import json
import signal
import logging
from typing import Any, Dict, List, Optional


#############################################
# Logging Setup
#############################################
logger = logging.getLogger("AURA.Core")
if not logger.handlers:
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	)


#############################################
# Environment Flags & Helpers
#############################################
def env_bool(name: str, default: bool = False) -> bool:
	val = os.getenv(name)
	if val is None:
		return default
	return val.strip().lower() in {"1", "true", "yes", "on"}


HEADLESS = env_bool("HEADLESS", True)
MINIMAL_MODE = env_bool("MINIMAL_MODE", True)
RUN_DURATION = int(os.getenv("RUN_DURATION", "0"))  # 0 = unlimited
HEADLESS_COMMANDS = os.getenv("HEADLESS_COMMANDS", "")


#############################################
# Fallback / Minimal Implementations
#############################################
class MinimalNLPEngine:
	"""Very small NLP stub to keep pipeline running in constrained mode."""

	def analyze(self, text: str) -> Dict[str, Any]:
		sentiment = "positive" if any(w in text.lower() for w in ["good", "great", "awesome"]) else "neutral"
		return {
			"text": text,
			"intent": "emergency" if "emergency" in text.lower() else "generic",
			"emotion": sentiment,
			"sentiment": {"label": sentiment.upper(), "score": 0.75 if sentiment == "positive" else 0.5},
			"language": "en",
		}


class InMemoryKnowledgeGraph:
	"""Fallback memory store when Neo4j or heavy dependencies are unavailable."""

	def __init__(self):
		self._items: List[Dict[str, Any]] = []

	def store_context(self, data: Dict[str, Any]) -> str:
		data = dict(data)
		data["_id"] = str(len(self._items) + 1)
		self._items.append(data)
		return data["_id"]

	def query_by_emotion(self, emotion: str) -> List[Dict[str, Any]]:
		return [x for x in self._items if x.get("emotion") == emotion][-10:]


class SilentVoice:
	def speak(self, *_args, **_kwargs):
		pass


class NoVision:
	def process(self):  # placeholder
		return {"status": "vision_disabled"}


#############################################
# Dynamic Imports with Safe Fallbacks
#############################################
def safe_import(name: str):
	try:
		module = __import__(name)
		return module
	except Exception as e:
		logger.warning(f"Module '{name}' unavailable or failed: {e}")
		return None


#############################################
# Orchestrator
#############################################
class AURAOrchestrator:
	"""Core orchestrator coordinating subsystems with graceful degradation."""

	def __init__(self,
				 minimal: bool = MINIMAL_MODE,
				 headless: bool = HEADLESS,
				 run_duration: int = RUN_DURATION,
				 headless_commands: str = HEADLESS_COMMANDS):
		self.minimal = minimal
		self.headless = headless
		self.run_duration = run_duration
		self._start_time = time.time()
		self._shutdown = False
		self._heartbeat_interval = 10

		# Signal handling
		signal.signal(signal.SIGINT, self._signal_handler)
		signal.signal(signal.SIGTERM, self._signal_handler)

		logger.info(f"Starting AURAOrchestrator (minimal={self.minimal}, headless={self.headless})")

		self.voice = SilentVoice()
		self.vision = NoVision()
		self.web = None
		self.memory = None
		self.security = None
		self.access = None
		self.nlp = None

		self._init_services()

		# Prepare headless command queue
		self.command_queue: List[str] = []
		if self.headless and headless_commands:
			# Accept JSON array or comma separated
			headless_commands = headless_commands.strip()
			try:
				if headless_commands.startswith("["):
					self.command_queue = json.loads(headless_commands)
				else:
					self.command_queue = [c.strip() for c in headless_commands.split(",") if c.strip()]
			except Exception as e:
				logger.error(f"Failed to parse HEADLESS_COMMANDS: {e}")
		logger.info(f"Headless command queue: {self.command_queue}")

	# ---------------- Lifecycle -----------------
	def _signal_handler(self, signum, _frame):
		logger.info(f"Signal {signum} received; initiating shutdown.")
		self._shutdown = True

	def _init_services(self):
		# NLP
		if self.minimal:
			self.nlp = MinimalNLPEngine()
		else:
			nlp_module = safe_import("nlp_processor")
			if nlp_module and hasattr(nlp_module, "EnhancedNLPEngine"):
				try:
					self.nlp = nlp_module.EnhancedNLPEngine()
				except Exception as e:
					logger.error(f"Full NLP init failed, using minimal stub: {e}")
					self.nlp = MinimalNLPEngine()
			else:
				self.nlp = MinimalNLPEngine()

		# Memory
		if self.minimal:
			self.memory = InMemoryKnowledgeGraph()
		else:
			mm = safe_import("memory_manager")
			if mm and hasattr(mm, "KnowledgeGraph"):
				try:
					self.memory = mm.KnowledgeGraph()
				except Exception as e:
					logger.error(f"Neo4j memory unavailable ({e}); using in-memory fallback")
					self.memory = InMemoryKnowledgeGraph()
			else:
				self.memory = InMemoryKnowledgeGraph()

		# Security
		sec = safe_import("security")
		if sec and hasattr(sec, "SecurityHandler"):
			try:
				self.security = sec.SecurityHandler(security_level="medium")
			except Exception as e:
				logger.warning(f"Security handler init failed: {e}")
		# Access (system control)
		access_mod = safe_import("access")
		if access_mod and hasattr(access_mod, "AuraAssistant"):
			try:
				self.access = access_mod.AuraAssistant()
			except Exception as e:
				logger.warning(f"AuraAssistant init failed: {e}")

		# Web controller (optional; heavy)
		if not self.minimal:
			web_mod = safe_import("neuro_web_controller")
			if web_mod and hasattr(web_mod, "NeuroWebController"):
				try:
					self.web = web_mod.NeuroWebController(voice_interface=self.voice)
				except Exception as e:
					logger.warning(f"Web controller init failed: {e}")

	# ---------------- Command Processing -----------------
	def process_command(self, command: str) -> Dict[str, Any]:
		command = command.strip()
		logger.info(f"Processing command: {command}")

		# Emergency
		if "emergency" in command.lower():
			return self._handle_emergency(command)

		# Prefixed routing
		if command.startswith("web:") and self.web:
			query = command[len("web:"):].strip()
			return {"web_search": query}
		if command.startswith("memory store:"):
			text = command.split(":", 1)[1].strip()
			stored_id = self.memory.store_context({
				"text": text,
				"emotion": "neutral",
				"intent": "store",
			}) if self.memory else None
			return {"stored": stored_id}
		if command.startswith("memory emotion:"):
			emotion = command.split(":", 1)[1].strip()
			results = self.memory.query_by_emotion(emotion) if self.memory else []
			return {"results": results}

		# Generic: run NLP analysis then store
		analysis = self.nlp.analyze(command) if self.nlp else {"text": command}
		if self.memory:
			try:
				self.memory.store_context({
					"text": analysis.get("text", command),
					"emotion": analysis.get("emotion", "neutral"),
					"intent": analysis.get("intent", "generic"),
					"sentiment": analysis.get("sentiment"),
					"language": analysis.get("language", "en"),
				})
			except Exception as e:
				logger.warning(f"Failed to store context: {e}")
		return analysis

	def _handle_emergency(self, command: str) -> Dict[str, Any]:
		logger.warning(f"Emergency detected in command: {command}")
		# Hook for future escalation (notifications, system actions)
		return {"status": "emergency_acknowledged"}

	# ---------------- Main Loop -----------------
	def run(self):
		logger.info("AURAOrchestrator entering run loop")
		next_heartbeat = time.time() + self._heartbeat_interval
		try:
			while not self._shutdown:
				# Heartbeat
				now = time.time()
				if now >= next_heartbeat:
					logger.info("Heartbeat: alive | memory_items=%s", getattr(self.memory, '_items', '__'))
					next_heartbeat = now + self._heartbeat_interval

				# Run duration enforcement
				if self.run_duration and (now - self._start_time) >= self.run_duration:
					logger.info("Run duration reached; shutting down")
					break

				# Command acquisition
				if self.headless:
					if self.command_queue:
						cmd = self.command_queue.pop(0)
						self.process_command(cmd)
					else:
						time.sleep(0.5)
				else:
					try:
						cmd = input("AURA> ").strip()
						if cmd.lower() in {"exit", "quit"}:
							break
						if cmd:
							self.process_command(cmd)
					except EOFError:
						break
		finally:
			self.shutdown()

	# ---------------- Shutdown -----------------
	def shutdown(self):
		if self._shutdown:
			logger.info("Shutdown already in progress")
		self._shutdown = True
		logger.info("Shutting down AURAOrchestrator")
		# Cleanup web controller if present
		try:
			if self.web and hasattr(self.web, "cleanup"):
				self.web.cleanup()
		except Exception as e:
			logger.warning(f"Web controller cleanup failed: {e}")
		logger.info("Shutdown complete")


#############################################
# Entrypoint
#############################################
def main():
	orch = AURAOrchestrator()
	orch.run()


if __name__ == "__main__":
	main()

