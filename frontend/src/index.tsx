import React, { useEffect } from 'react'
import ReactDOM from 'react-dom/client'
import { HashRouter, Route, Routes } from 'react-router-dom'
import { initializeIcons } from '@fluentui/react'

import Chat from './pages/chat/Chat'
import Layout from './pages/layout/Layout'
import NoPage from './pages/NoPage'
import { AppStateProvider } from './state/AppProvider'

import './index.css'

initializeIcons(
    'https://res.cdn.office.net/files/fabric-cdn-prod_20240129.001/assets/icons/',
);

export default function App() {

    useEffect(() => {
        fetch("/log_user_access", { method: "GET", credentials: "include" })
            .then(response => response.json())
            .then(data => console.log("User access logged:", data))
            .catch(error => console.error("Logging user access failed:", error));
    }, []); // Empty dependency array ensures it runs once on mount

    return (
        <AppStateProvider>
            <HashRouter>
                <Routes>
                    <Route path="/" element={<Layout />}>
                        <Route index element={<Chat />} />
                        <Route path="*" element={<NoPage />} />
                    </Route>
                </Routes>
            </HashRouter>
        </AppStateProvider>
    )
}

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
)
