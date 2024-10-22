// src/components/ProtectedRoute.js
import React from 'react';
import { Route, Navigate } from 'react-router-dom';

const ProtectedRoute = ({ element, ...rest }) => {
    const isAuthenticated = !!localStorage.getItem("access_token"); // Check if token exists

    return (
        <Route
            {...rest}
            element={isAuthenticated ? element : <Navigate to="/" replace />}
        />
    );
};

export default ProtectedRoute;
