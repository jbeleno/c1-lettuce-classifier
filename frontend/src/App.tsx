import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Predict } from './routes/Predict'
import { History } from './routes/History'
import { Metrics } from './routes/Metrics'
import { Models } from './routes/Models'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Predict />} />
          <Route path="history" element={<History />} />
          <Route path="metrics" element={<Metrics />} />
          <Route path="models" element={<Models />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
