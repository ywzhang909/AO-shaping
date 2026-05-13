"""Port listening utility for testing R50Controller protocol communication.

This utility can be used to listen on a TCP port and capture the raw bytes
sent by the R50Controller to verify protocol compliance during testing.
"""

from __future__ import annotations

import socket
import threading
import time
import logging
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CapturedCommand:
    """Represents a captured command from the R50Controller."""
    data: bytes
    timestamp: float
    peer_info: str


class PortListener:
    """TCP server that listens for connections and captures all data sent.
    
    This utility is useful for testing the R50Controller by allowing you to
    verify that the correct protocol bytes are being sent over the wire.
    
    Example usage:
        listener = PortListener(port=10101)
        listener.start()
        # Connect your R50Controller to localhost:10101
        controller = R50Controller(1, "127.0.0.1", 10101)
        controller.connect()
        
        # Send some commands
        controller.set_all_channel_voltage(5.0)
        controller.set_channel_voltage(0, 10.0)
        
        # Check what was actually sent
        commands = listener.get_captured_commands()
        assert len(commands) == 2
        # Verify the protocol format of each command
        listener.stop()
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        """Initialize the port listener.
        
        Args:
            host: Host address to bind to (default: 127.0.0.1)
            port: Port to listen on (default: 0 for random available port)
        """
        self.host = host
        self.port = port
        self._server_socket: Optional[socket.socket] = None
        self._captured_commands: List[CapturedCommand] = []
        self._client_sockets: List[socket.socket] = []
        self._is_running = False
        self._listen_thread: Optional[threading.Thread] = None
        
    def start(self) -> None:
        """Start the TCP listener."""
        if self._is_running:
            return
            
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(5)
        
        # Get the actual port if we used port 0
        if self.port == 0:
            self.port = self._server_socket.getsockname()[1]
            
        self._is_running = True
        self._listen_thread = threading.Thread(target=self._listen_for_connections, daemon=True)
        self._listen_thread.start()
        logger.info(f"PortListener started on {self.host}:{self.port}")
        
    def stop(self) -> None:
        """Stop the TCP listener and cleanup."""
        self._is_running = False
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
            self._server_socket = None
            
        # Close all client connections
        for sock in self._client_sockets:
            try:
                sock.close()
            except Exception:
                pass
        self._client_sockets.clear()
        
        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=1)
            
        logger.info("PortListener stopped")
        
    def _listen_for_connections(self) -> None:
        """Listen for incoming connections and handle them."""
        while self._is_running:
            try:
                self._server_socket.settimeout(1.0)  # 1 second timeout to allow checking _is_running
                client_socket, address = self._server_socket.accept()
                if not self._is_running:
                    break
                    
                peer_info = f"{address[0]}:{address[1]}"
                logger.debug(f"New connection from {peer_info}")
                
                self._client_sockets.append(client_socket)
                
                # Handle client in a separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, peer_info),
                    daemon=True
                )
                client_thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                if self._is_running:
                    logger.warning(f"Error accepting connection: {e}")
                break
                
    def _handle_client(self, client_socket: socket.socket, peer_info: str) -> None:
        """Handle a client connection and capture all data sent."""
        try:
            while self._is_running:
                # Read data until connection closes
                data = client_socket.recv(1024)  # Read in chunks
                if not data:
                    break
                    
                # Capture the data with timestamp
                captured = CapturedCommand(
                    data=data,
                    timestamp=time.time(),
                    peer_info=peer_info
                )
                self._captured_commands.append(captured)
                logger.debug(f"Captured {len(data)} bytes from {peer_info}: {data.hex()}")
                
        except Exception as e:
            logger.debug(f"Error handling client {peer_info}: {e}")
        finally:
            logger.debug(f"Connection from {peer_info} closed")
            try:
                client_socket.close()
            except Exception:
                pass
            if client_socket in self._client_sockets:
                self._client_sockets.remove(client_socket)
                
    def get_captured_commands(self) -> List[CapturedCommand]:
        """Get all captured commands since the listener started.
        
        Returns:
            List of CapturedCommand objects in chronological order
        """
        return self._captured_commands.copy()
        
    def clear_captured_commands(self) -> None:
        """Clear the captured commands buffer."""
        self._captured_commands.clear()
        
    def get_raw_bytes(self) -> bytes:
        """Get all captured data concatenated as raw bytes.
        
        Returns:
            Concatenated bytes of all captured data
        """
        return b''.join(cmd.data for cmd in self._captured_commands)
        
    def get_command_count(self) -> int:
        """Get the number of captured command chunks.
        
        Returns:
            Number of times data was received (may be multiple reads per logical command)
        """
        return len(self._captured_commands)