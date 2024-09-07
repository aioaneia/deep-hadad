
import './App.css';

import {
    MetricsPerEpochChart,
    NetworksComparisonChart,
    LossFunctionsChart,
    NormalizedMetricsChart,
} from './MetricsChart';

// import GeneratorDiagram from "./Network";

function App() {
    return (
        <div className="App">
            <MetricsPerEpochChart />
            <NetworksComparisonChart />
            <LossFunctionsChart />
            <NormalizedMetricsChart />

            {/*<GeneratorDiagram />*/}
        </div>
    );
}

export default App;
