import { useState } from "react";
import "./index.css";
import { useNavigate } from "react-router-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faEye, faEyeSlash } from "@fortawesome/free-solid-svg-icons";

export default function LoginPage() {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const navigate = useNavigate();
    const [showPassword, setShowPassword] = useState(false);
    const [error, setError] = useState(null);

    const handleEmailChange = (event) => {
        setEmail(event.target.value);
    };

    const handlePasswordChange = (event) => {
        setPassword(event.target.value);
    };

    const handlePasswordToggle = () => {
        setShowPassword(!showPassword);
    };

    const handleLogin = async (event) => {
        event.preventDefault();
        setError(null); // Reset error before new login attempt
        try {
            const response = await fetch("http://127.0.0.1:8000/token", {
                method: "POST",
                headers: {
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                body: new URLSearchParams({
                    username: email,
                    password: password,
                }).toString(),
            });
    
            const resp = await response.json();
            if (response.ok) {
                localStorage.setItem("access_token", resp.access_token);
                navigate("/interview");
            } else {
                throw new Error(resp.detail || "Failed to login");
            }
        } catch (error) {
            setError(error.message);
        }
    };
    

    return (
        <div className="login-box">
            <div className="left-container">
                <form onSubmit={handleLogin}>
                    <h2 id="login-cotation">
                        Login in to Crack Your Interview
                    </h2>
                    <p>
                        Choose a job you love, and you will never have to work a
                        day in your life.
                    </p>
                    <div className="form-box">
                        <label htmlFor="email">Email Id</label>
                        <input
                            type="email" // Change to email for better validation
                            id="email" // Added id for accessibility
                            onChange={handleEmailChange}
                            value={email}
                            placeholder="Email"
                            required // Add required attribute
                        />
                        <label htmlFor="password">Password</label>
                        <input
                            type={showPassword ? "text" : "password"}
                            id="password" // Added id for accessibility
                            onChange={handlePasswordChange}
                            value={password}
                            placeholder="Password"
                            required // Add required attribute
                        />
                        <FontAwesomeIcon
                            id="toggle-Password"
                            icon={showPassword ? faEye : faEyeSlash}
                            onClick={handlePasswordToggle}
                            style={{ cursor: "pointer", marginLeft: "5px" }}
                        />
                        <button id="login-button">Login</button>
                    </div>
                    {error && <p className="error">{error}</p>}{" "}
                    {/* Display error message if error exists */}
                </form>
            </div>
            <div className="right-container"></div>
        </div>
    );
}
