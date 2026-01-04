import React, { useState } from 'react';

const ValueRenderer =({ value, level = 0 })=> {
    const [open, setOpen] = React.useState(false);

    const indent = level;

    /* ---------- ARRAY ---------- */
    if (Array.isArray(value)) {
        return (
            <div style={{marginLeft: indent,color:'black' }} className="max-w-[700px] content-start">
                <button
                    onClick={() => setOpen(!open)}
                    className="text-gray-500 flex-shrink-0 select-none  hover:underline"
                >
                    Array ({value.length})
                </button>

                {open && (
                    <div className="mt-2 space-y-1 border-l border-gray-300 pl-2">
                        {value.map((item, idx) => (
                            <div key={idx} className="flex  ">
                                <span className="text-gray-500 text-left  select-none">
                                    [{idx}]:
                                </span>
                                <ValueRenderer value={item} level={level + 1} />
                            </div>
                        ))}
                    </div>
                )}
            </div>
        );
    }

    /* ---------- OBJECT ---------- */
    if (typeof value === "object" && value !== null) {
        return (
            <div style={{ marginLeft: indent }} className="max-w-[700px]">
                <button
                    onClick={() => setOpen(!open)}
                    className="text-gray-500 flex-shrink-0 select-none hover:underline"
                >
                    Object
                </button>

                {open && (
                    <div className="mt-2 space-y-1 border-l border-gray-300 pl-2  align-items: flex-start">
                        {Object.entries(value).map(([k, v]) => (
                            <div key={k} className="flex gap-2">
                                <span className="text-gray-500  text-left select-none">
                                    {k} :
                                </span>
                                <ValueRenderer value={v} level={level + 1} />
                            </div>
                        ))}
                    </div>
                )}
            </div>
        );
    }

    // /* ---------- PRIMITIVES ---------- */
    // if (typeof value === "string") {
    //     return (
    //         <span className="text-green-700 break-all max-w-[700px] inline-block">
    //             "{value}"
    //         </span>
    //     );
    // }

    // if (typeof value === "number") {
    //     return <span className="text-blue-600">{value}</span>;
    // }

    // if (typeof value === "boolean") {
    //     return <span className="text-purple-600">{String(value)}</span>;
    // }

    // if (value === null) {
    //     return <span className="text-gray-500">null</span>;
    // }

    // return <span className="text-gray-800">{String(value)}</span>;
    return value;
}

export default ValueRenderer;