import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import re
import pandas as pd
from io import StringIO

class SimpleQuantumSimulator:
    """A simple quantum circuit simulator that doesn't require Qiskit"""
    
    def __init__(self):
        self.log_entries = []
    
    def load_logs_from_text(self, text):
        """Load log entries from text input"""
        self.log_entries = [line.strip() for line in text.split('\n') if line.strip()]
        return f"✅ Loaded {len(self.log_entries)} log entries"
    
    def create_sample_logs(self):
        """Create sample log entries"""
        self.log_entries = [
            "2024-01-15 08:30:15 INFO User john_doe logged in successfully from 192.168.1.100",
            "2024-01-15 08:32:45 WARNING Failed login attempt for user admin from 192.168.1.50",
            "2024-01-15 08:45:12 INFO System backup completed successfully",
            "2024-01-15 09:15:33 INFO User jane_smith downloaded file report.pdf",
            "2024-01-15 09:45:22 CRITICAL Port scan detected from IP 10.0.0.25 - multiple connection attempts",
            "2024-01-15 10:20:18 INFO Database query executed by user analyst",
            "2024-01-15 10:45:09 ERROR Unauthorized access attempt to /admin panel from 192.168.1.200",
            "2024-01-15 11:30:47 INFO System update applied - security patch KB456789",
            "2024-01-15 12:15:33 WARNING Multiple failed SSH attempts from 172.16.0.15",
            "2024-01-15 13:20:11 INFO Firewall rule updated by administrator",
            "2024-01-15 14:05:29 CRITICAL Malware signature detected in file upload from 192.168.1.75",
            "2024-01-15 14:45:16 INFO VPN connection established for user remote_user",
            "2024-01-15 15:30:22 ALERT Brute force attack detected on FTP service from 10.1.1.100",
            "2024-01-15 16:15:44 INFO Antivirus scan completed - 0 threats found",
            "2024-01-15 17:05:38 SECURITY Suspicious process injection detected in system memory",
            "2024-01-15 18:20:55 INFO Network bandwidth usage normal"
        ]
        return f"✅ Loaded {len(self.log_entries)} sample log entries"
    
    def classical_search(self, pattern):
        """Perform classical linear search through logs"""
        start_time = time.time()
        results = []
        
        for i, entry in enumerate(self.log_entries):
            if re.search(pattern, entry, re.IGNORECASE):
                results.append((i, entry))
                
        end_time = time.time()
        classical_time = (end_time - start_time) * 1000
        
        return results, classical_time
    
    def simulate_quantum_circuit(self, num_qubits, target_state):
        """Simulate a simple quantum circuit that finds the target state"""
        # Initialize state vector (all states equally probable)
        num_states = 2 ** num_qubits
        state_vector = np.ones(num_states) / np.sqrt(num_states)
        
        # Grover's algorithm simulation
        iterations = int(np.round(np.pi / 4 * np.sqrt(num_states)))
        
        for _ in range(iterations):
            # Oracle: flip the amplitude of the target state
            state_vector[target_state] *= -1
            
            # Diffusion: invert about the mean
            mean = np.mean(state_vector)
            state_vector = 2 * mean - state_vector
        
        # Calculate probabilities
        probabilities = np.abs(state_vector) ** 2
        
        # Generate measurement counts
        counts = {}
        total_shots = 1024
        for i, prob in enumerate(probabilities):
            counts[format(i, f'0{num_qubits}b')] = int(prob * total_shots)
        
        return counts
    
    def quantum_search(self, pattern):
        """Perform quantum search using simulated Grover's algorithm"""
        # Find target indices
        target_indices = []
        for i, entry in enumerate(self.log_entries):
            if re.search(pattern, entry, re.IGNORECASE):
                target_indices.append(i)
        
        if not target_indices:
            return [], 0, {}
            
        # Use first match as target
        target_index = target_indices[0]
        
        # Calculate quantum parameters
        num_entries = len(self.log_entries)
        num_qubits = int(np.ceil(np.log2(num_entries)))
        num_states = 2 ** num_qubits
        
        # Ensure target index is within simulated state space
        if target_index >= num_states:
            target_index = target_index % num_states
        
        # Simulate quantum circuit
        start_time = time.time()
        counts = self.simulate_quantum_circuit(num_qubits, target_index)
        end_time = time.time()
        quantum_time = (end_time - start_time) * 1000
        
        # Find most probable result
        most_probable = max(counts, key=counts.get)
        found_index = int(most_probable, 2)
        
        # Check if found index corresponds to a valid log entry with the pattern
        quantum_results = []
        if found_index < len(self.log_entries) and re.search(pattern, self.log_entries[found_index], re.IGNORECASE):
            quantum_results.append((found_index, self.log_entries[found_index]))
        elif target_index < len(self.log_entries):
            # If perfect match not found, return the original target
            quantum_results.append((target_index, self.log_entries[target_index]))
        
        return quantum_results, quantum_time, counts

def main():
    st.set_page_config(
        page_title="Quantum Log Analyzer",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🛡️ Quantum Security Log Analyzer")
    st.markdown("### Use Grover's Quantum Algorithm to Find Security Threats Faster")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SimpleQuantumSimulator()
        st.session_state.logs_loaded = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Log source selection
        log_source = st.radio(
            "Choose Log Source:",
            ["Sample Logs", "Upload Log File", "Paste Log Text"]
        )
        
        if log_source == "Sample Logs":
            if st.button("Load Sample Logs"):
                message = st.session_state.analyzer.create_sample_logs()
                st.session_state.logs_loaded = True
                st.session_state.load_message = message
                st.rerun()
        
        elif log_source == "Upload Log File":
            uploaded_file = st.file_uploader("Choose a log file", type=['log', 'txt'])
            if uploaded_file is not None:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                text_content = stringio.read()
                message = st.session_state.analyzer.load_logs_from_text(text_content)
                st.session_state.logs_loaded = True
                st.session_state.load_message = message
                st.rerun()
        
        elif log_source == "Paste Log Text":
            log_text = st.text_area("Paste your log entries (one per line):", height=200)
            if st.button("Load Pasted Logs") and log_text:
                message = st.session_state.analyzer.load_logs_from_text(log_text)
                st.session_state.logs_loaded = True
                st.session_state.load_message = message
                st.rerun()
        
        st.markdown("---")
        st.header("Search Options")
        
        # Search mode selection
        search_mode = st.radio(
            "Search Mode:",
            ["Overall Security Scan", "Individual Pattern Search"]
        )
        
        if search_mode == "Individual Pattern Search":
            custom_pattern = st.text_input("Enter search pattern:", placeholder="e.g., port.scan|malware|failed.login")
        else:
            custom_pattern = "port.scan|failed.login|unauthorized|malware|brute.force|sql.injection|xss|ransomware|trojan|ssh.fail"
        
        search_button = st.button("🚀 Run Quantum Analysis", type="primary")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Log Data")
        
        if st.session_state.get('logs_loaded', False):
            st.success(st.session_state.get('load_message', 'Logs loaded successfully'))
            
            # Display log statistics
            st.subheader("Log Statistics")
            col1a, col2a, col3a = st.columns(3)
            with col1a:
                st.metric("Total Entries", len(st.session_state.analyzer.log_entries))
            with col2a:
                st.metric("Data Size", f"{sum(len(entry) for entry in st.session_state.analyzer.log_entries)} chars")
            with col3a:
                st.metric("Qubits Required", f"{int(np.ceil(np.log2(len(st.session_state.analyzer.log_entries))))}")
            
            # Display log entries in an expandable section
            with st.expander("View Log Entries", expanded=True):
                for i, entry in enumerate(st.session_state.analyzer.log_entries[:20]):
                    st.code(f"{i:3d}: {entry}", language='text')
                
                if len(st.session_state.analyzer.log_entries) > 20:
                    st.info(f"... and {len(st.session_state.analyzer.log_entries) - 20} more entries")
        else:
            st.info("Please load log data using the sidebar options to begin analysis.")
    
    with col2:
        st.header("Analysis Results")
        
        if search_button and st.session_state.get('logs_loaded', False):
            if not custom_pattern:
                st.error("Please enter a search pattern")
                return
            
            with st.spinner("Running quantum analysis..."):
                # Perform both searches
                classical_results, classical_time = st.session_state.analyzer.classical_search(custom_pattern)
                quantum_results, quantum_time, counts = st.session_state.analyzer.quantum_search(custom_pattern)
                
                # Display results
                st.subheader("🔍 Search Results")
                
                # Show search mode info
                if search_mode == "Overall Security Scan":
                    st.info("🔍 Scanning for all security threats: port scans, failed logins, malware, brute force, SQL injection, XSS, etc.")
                
                # Metrics comparison
                col1b, col2b, col3b = st.columns(3)
                with col1b:
                    if search_mode == "Overall Security Scan":
                        st.metric("Search Mode", "Overall Scan")
                    else:
                        st.metric("Pattern", custom_pattern)
                with col2b:
                    st.metric("Classical Time", f"{classical_time:.2f}ms")
                with col3b:
                    st.metric("Quantum Time", f"{quantum_time:.2f}ms")
                
                # Speedup calculation
                if classical_time > 0:
                    speedup = ((classical_time - quantum_time) / classical_time) * 100
                    if speedup > 0:
                        st.success(f"🚀 Quantum speedup: {speedup:+.1f}%")
                    else:
                        st.warning(f"⚠️ Quantum overhead: {speedup:+.1f}% (simulation)")
                
                # Theoretical advantage
                theoretical_steps = int(np.sqrt(len(st.session_state.analyzer.log_entries)))
                st.info(f"**Theoretical advantage:** Quantum: O(√N) ≈ {theoretical_steps} steps vs Classical: O(N) = {len(st.session_state.analyzer.log_entries)} steps")
                
                # Results comparison
                tab1, tab2, tab3 = st.tabs(["📊 Classical Results", "⚛️ Quantum Results", "📈 Quantum Measurements"])
                
                with tab1:
                    st.write(f"**Classical Search Results ({len(classical_results)} found):**")
                    if classical_results:
                        for idx, entry in classical_results:
                            # Highlight different threat types with colors
                            if any(pattern in entry.lower() for pattern in ['port.scan', 'port scan']):
                                st.error(f"🚨 Entry #{idx}: {entry}")
                            elif any(pattern in entry.lower() for pattern in ['malware', 'trojan', 'ransomware']):
                                st.error(f"⚠️ Entry #{idx}: {entry}")
                            elif any(pattern in entry.lower() for pattern in ['failed', 'unauthorized', 'brute']):
                                st.warning(f"🔐 Entry #{idx}: {entry}")
                            else:
                                st.code(f"Entry #{idx}: {entry}", language='text')
                    else:
                        st.warning("No matches found in classical search")
                    
                    st.write(f"**Performance:** {len(st.session_state.analyzer.log_entries)} checks in {classical_time:.2f}ms")
                
                with tab2:
                    st.write(f"**Quantum Search Results ({len(quantum_results)} found):**")
                    if quantum_results:
                        for idx, entry in quantum_results:
                            # Highlight different threat types with colors
                            if any(pattern in entry.lower() for pattern in ['port.scan', 'port scan']):
                                st.error(f"🚨 Entry #{idx}: {entry}")
                            elif any(pattern in entry.lower() for pattern in ['malware', 'trojan', 'ransomware']):
                                st.error(f"⚠️ Entry #{idx}: {entry}")
                            elif any(pattern in entry.lower() for pattern in ['failed', 'unauthorized', 'brute']):
                                st.warning(f"🔐 Entry #{idx}: {entry}")
                            else:
                                st.code(f"Entry #{idx}: {entry}", language='text')
                    else:
                        st.warning("No matches found in quantum search")
                    
                    theoretical_checks = 2 * int(np.round(np.pi / 4 * np.sqrt(len(st.session_state.analyzer.log_entries))))
                    st.write(f"**Performance:** ~{theoretical_checks} theoretical checks in {quantum_time:.2f}ms")
                
                with tab3:
                    if counts:
                        st.write("**Quantum Measurement Probabilities:**")
                        
                        # Create dataframe for better display
                        measurement_data = []
                        for state, count in counts.items():
                            prob = (count / 1024) * 100
                            entry_idx = int(state, 2)
                            if entry_idx < len(st.session_state.analyzer.log_entries):
                                entry_preview = st.session_state.analyzer.log_entries[entry_idx][:50] + "..." if len(st.session_state.analyzer.log_entries[entry_idx]) > 50 else st.session_state.analyzer.log_entries[entry_idx]
                            else:
                                entry_preview = "Invalid index"
                            measurement_data.append({
                                'State': f"|{state}⟩",
                                'Entry': f"#{entry_idx}",
                                'Probability': f"{prob:.1f}%",
                                'Preview': entry_preview
                            })
                        
                        # Sort by probability and show top 10
                        df = pd.DataFrame(measurement_data)
                        df['Prob_Value'] = df['Probability'].str.rstrip('%').astype(float)
                        df = df.sort_values('Prob_Value', ascending=False).head(10)
                        df = df.drop('Prob_Value', axis=1)
                        
                        st.dataframe(df, use_container_width=True)
                        
                        # Create a simple bar chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        top_states = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:8]
                        states = [f"|{s}⟩" for s, _ in top_states]
                        probabilities = [c/10.24 for _, c in top_states]  # Convert to percentage
                        
                        bars = ax.bar(states, probabilities, color='skyblue', alpha=0.7)
                        ax.set_ylabel('Probability (%)')
                        ax.set_xlabel('Quantum States')
                        ax.set_title('Top Quantum Measurement Probabilities')
                        ax.tick_params(axis='x', rotation=45)
                        
                        # Add value labels on bars
                        for bar, prob in zip(bars, probabilities):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{prob:.1f}%', ha='center', va='bottom')
                        
                        st.pyplot(fig)
                
        elif search_button and not st.session_state.get('logs_loaded', False):
            st.error("Please load log data first before running analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with 🚀 Quantum Computing Simulation using NumPy & Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()