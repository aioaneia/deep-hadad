import React from 'react';

const NetworkDiagram = () => (
    <svg width="1200" height="1200" viewBox="0 0 1200 1200">
        <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#555"/>
            </marker>
        </defs>

        {/* Generator */}
        <text x="555" y="25" fontSize="24" fontWeight="bold" textAnchor="middle" fontStyle="italic">DeepHadad</text>
        <text x="685" y="25" fontSize="24" fontWeight="bold" textAnchor="middle">Generator</text>

        <rect x="50" y="40" width="1115" height="245" fill="#e5e7eb" stroke="#d1d5db" strokeWidth="2"/>

        {/* Generator layers */}
        {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17].map((i) => (
            <g key={`gen-${i}`}>
                {/* Column Layers */}
                <rect x={70 + i * 60} y="70" width="55" height="200" fill="#e5e7eb" stroke="#d1d5db" strokeWidth="0"/>

                {/* Initial layer */}
                {
                    i === 0 && (
                        <>
                            <rect x={70 + i * 60} y="70" width="55" height="50" fill="#beae9c"/>
                            {/* ReflectionPad2d */}
                            <rect x={70 + i * 60} y="120" width="55" height="50" fill="#9975e5"/>
                            {/* Conv2D */}
                            <rect x={70 + i * 60} y="170" width="55" height="50" fill="#bfdbfe"/>
                            {/* InstanceNorm2d */}
                            <rect x={70 + i * 60} y="220" width="55" height="50" fill="#6e85f3"/>
                            {/* ReLU */}
                        </>
                    )
                }

                {/* Downsampling Layers */}
                {
                    i > 0 && i < 4 && (
                        <>
                            <rect x={70 + i * 60} y="70" width="55" height="50" fill="#9975e5"/>
                            {/* Conv2D */}
                            <rect x={70 + i * 60} y="120" width="55" height="50" fill="#bfdbfe"/>
                            {/* InstanceNorm2d */}
                            <rect x={70 + i * 60} y="170" width="55" height="50" fill="#6e85f3"/>
                            {/* ReLU */}
                        </>
                    )
                }

                {/* Self-Attention layer */}
                {
                    i === 4 && <rect x={70 + i * 60} y="70" width="55" height="150" fill="#fde68a"/>
                }

                {/* SPADE ResBlock Layers */}
                {
                    i > 4 && i < 14 && <rect x={70 + i * 60} y="70" width="55" height="50" fill="#6ee7b7"/>
                }

                {/* SPADE Module */}
                {
                    i > 4 && i < 14 && <rect x={70 + i * 60} y="120" width="55" height="50" fill="#fdba74"/>
                }

                {/* Upsampling Layers */}
                {
                    i > 13 && i < 17 && (
                        <>
                            <rect x={70 + i * 60} y="70" width="55" height="50" fill="#b77474"/>
                            {/* ConvTranspose2d */}
                            <rect x={70 + i * 60} y="120" width="55" height="50" fill="#bfdbfe"/>
                            {/* InstanceNorm2d */}
                            <rect x={70 + i * 60} y="170" width="55" height="50" fill="#6e85f3"/>
                            {/* ReLU */}
                        </>
                    )
                }

                {/* Final layer */}
                {
                    i === 17 && (
                        <>
                            <rect x={70 + i * 60} y="70" width="55" height="50" fill="#beae9c"/>
                            {/* ReflectionPad2d */}
                            <rect x={70 + i * 60} y="120" width="55" height="50" fill="#9975e5"/>
                            {/* Conv2d */}
                            <rect x={70 + i * 60} y="170" width="55" height="50" fill="#747779"/>
                            {/* Sigmoid */}
                        </>
                    )
                }
            </g>
        ))}

        {
            /* Skip connections */
            /* layers 1, 2, 3, are connected to layers 16, 15, 14 respectively */
        }
        {[1, 2, 3].map((i) => (
            <path
                key={`skip-${i}`}
                d={
                    `M ${90 + i * 60} 70 C ${90 + i * 60} 40, ${930 + (i - 1) * 60} 40, ${930 + (i - 1) * 60} 70`
                }
                fill="none"
                stroke="#94a3b8"
                strokeWidth="2"
                strokeDasharray="5"
            />
        ))}

        {/* Discriminator title*/}
        <text x="380" y="530" fontSize="24" fontWeight="bold" textAnchor="middle" fontStyle="italic">DeepHadad</text>
        <text x="530" y="530" fontSize="24" fontWeight="bold" textAnchor="middle">Discriminator</text>

        {/* Layer columns */}
        <rect x="50" y="550" width="800" height="240" fill="#fff7ed" stroke="#d1d5db" strokeWidth="2"/>

        {/* Discriminator layers */}
        {[0, 1, 2, 3, 4, 5, 6].map((i) => (
            <g key={`dis-${i}`}>
                <rect x={70 + i * 110} y="570" width="100" height="200" fill="#fff7ed" stroke="#d1d5db"
                      strokeWidth="0"/>

                {/* First Layers */}
                {
                    i === 0 && (
                        <>
                            <rect x={70 + i * 110} y="570" width="100" height="50" fill="#9975e5"/>
                            {/* Conv2D */}
                            <rect x={70 + i * 110} y="620" width="100" height="50" fill="#505d9a"/>
                            {/* LeakyReLU */}
                        </>
                    )
                }

                {/* Intermediate Layers */}
                {
                    i > 0 && i < 6 && (
                        <>
                            <rect x={70 + i * 110} y="570" width="100" height="50" fill="#9975e5"/>
                            {/* Conv2D */}
                            <rect x={70 + i * 110} y="620" width="100" height="50" fill="#505d9a"/>
                            {/* LeakyReLU */}
                            <rect x={70 + i * 110} y="670" width="100" height="50" fill="#bfdbfe"/>
                            {/* InstanceNorm2d */}
                        </>
                    )
                }

                {/* Self-Attention Layers */}
                {
                    (i === 1 || i === 3) && <rect x={70 + i * 110} y="720" width="100" height="50" fill="#fde68a"/>
                }

                {/* Final Layer */}
                {
                    i === 6 && (
                        <>
                            <rect x={70 + i * 110} y="570" width="100" height="50" fill="#a0c566"/>
                            {/* ResBlock */}
                            <rect x={70 + i * 110} y="620" width="100" height="50" fill="#9975e5"/>
                            {/* Conv2d */}
                            <rect x={70 + i * 110} y="670" width="100" height="50" fill="#505d9a"/>
                            {/* LeakyReLU */}
                            <rect x={70 + i * 110} y="720" width="100" height="50" fill="#9975e5"/>
                            {/* Conv2d */}
                        </>
                    )
                }
            </g>
        ))}

        {/* Legend */}
        <text x="450" y="830" fontSize="20" fontWeight="bold" textAnchor="middle">Legend</text>
        <g transform="translate(50, 850)">
            {[
                {color: "#beae9c", label: "ReflectionPad2d"},
                {color: "#9975e5", label: "Conv2D"},
                {color: "#6e85f3", label: "ReLU"},
                {color: "#505d9a", label: "LeakyReLU"},
                {color: "#bfdbfe", label: "InstanceNorm2d"},
                {color: "#6ee7b7", label: "SPADE ResBlock"},
                {color: "#fdba74", label: "SPADE Module"},
                {color: "#b77474", label: "ConvTranspose2d"},
                {color: "#747779", label: "Sigmoid"},
                {color: "#fde68a", label: "Self-Attention"},
                {color: "#8ca664", label: "ResBlock"},
            ].map(({color, label}, i) => (
                <g key={label} transform={`translate(${(i % 6) * 139}, ${Math.floor(i / 6) * 20})`}>
                    <rect width="15" height="15" fill={color}/>
                    <text x="25" y="12" fontSize="13">{label}</text>
                </g>
            ))}
        </g>

        {/* Main flow arrow */}
        {/*<line x1="50" y1="220" x2="1150" y2="220" stroke="#555" strokeWidth="2" markerEnd="url(#arrowhead)"/>*/}
    </svg>
);


export default NetworkDiagram;