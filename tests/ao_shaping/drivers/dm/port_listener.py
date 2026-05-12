"""Port listening utility for testing R50Controller protocol communication.

This utility can be used to listen on a TCP port and capture the raw bytes
sent by the R50Controller to verify protocol compliance during testing.
"""

from __future__ import annotations

import asyncio
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
        async with PortListener(port=10101) as listener:
            # Connect your R50Controller to localhost:10101
            controller = R50Controller(1, "127.0.0.1", 10101)
            await controller.connect()
            
            # Send some commands
            await controller.set_all_channel_voltage(5.0)
            await controller.set_channel_voltage(0, 10.0)
            
            # Check what was actually sent
            commands = listener.get_captured_commands()
            assert len(commands) == 2
            # Verify the protocol format of each command
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        """Initialize the port listener.
        
        Args:
            host: Host address to bind to (default: 127.0.0.1)
            port: Port to listen on (default: 0 for random available port)
        """
        self.host = host
        self.port = port
        self._server: Optional[asyncio.Server] = None
        self._captured_commands: List[CapturedCommand] = []
        self._connections: List[asyncio.StreamReader] = []
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
        
    async def start(self) -> None:
        """Start the TCP listener."""
        self._server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port
        )
        # Get the actual port if we used port 0
        if self.port == 0:
            sock = self._server.sockets[0]
            self.port = sock.getsockname()[1]
        logger.info(f"PortListener started on {self.host}:{self.port}")
        
    async def stop(self) -> None:
        """Stop the TCP listener and cleanup."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            
        # Close all client connections
        for conn in self._connections:
            if not conn.at_eof():
                # Try to close gracefully
                pass
        self._connections.clear()
        logger.info("PortListener stopped")
        
    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle a client connection and capture all data sent."""
        peer_name = writer.get_extra_info('peername')
        peer_info = f"{peer_name[0]}:{peer_name[1]}" if peer_name else "unknown"
        logger.debug(f"New connection from {peer_info}")
        
        self._connections.append(reader)
        
        try:
            while True:
                # Read data until connection closes
                data = await reader.read(1024)  # Read in chunks
                if not data:
                    break
                    
                # Capture the data with timestamp
                import time
                captured = CapturedCommand(
                    data=data,
                    timestamp=time.time(),
                    peer_info=peer_info
                )
                self._captured_commands.append(captured)
                logger.debug(f"Captured {len(data)} bytes from {peer_info}: {data.hex()}")
                
        except asyncio.IncompleteReadError:
            # Connection closed by peer
            pass
        except Exception as e:
            logger.warning(f"Error handling client {peer_info}: {e}")
        finally:
            logger.debug(f"Connection from {peer_info} closed")
            if reader in self._connections:
                self._connections.remove(reader)
                
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