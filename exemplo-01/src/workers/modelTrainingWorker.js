import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';
let _globalCtx = {}
let _model = null
const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1
}

// Normalizar os valores de price e age para o range de 0-1
// Porque? Mantem todos os parametros balanceados para que nenhuma se sobressaia sobre outra
// Formula: (val - min) / (max - min)
// Exemplo: price=129.99, minPrice=39.99, maxPrice=199.99 → 0.56
const normalize = (value, min, max) => (value - min) / ((max - min) || 1)

function makeContext(products, users) {
    const ages = users.map(u => u.age)
    const prices = products.map(p => p.price)

    const minAge = Math.min(...ages)
    const maxAge = Math.max(...ages)

    const minPrice = Math.min(...prices)
    const maxPrice = Math.max(...prices)

    const colors = [...new Set(products.map(p => p.color))]
    const categories = [...new Set(products.map(p => p.category))]

    const colorsIndex = Object.fromEntries(
        colors.map((color, index) => {
            return [color, index]
        }))
    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => {
            return [category, index]
        }))

    // Computar a média de idade dos comprados por produto(ajuda a personalizar)
    const midAge = (minAge + maxAge) / 2
    const ageSums = {}
    const ageCounts = {}

    users.forEach(user => {
        user.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1
        })
    })

    const productAvgAgeNorm = Object.fromEntries(
        products.map(product => {
            const avg = ageCounts[product.name] ?
                ageSums[product.name] / ageCounts[product.name] :
                midAge

            return [product.name, normalize(avg, minAge, maxAge)]
        })
    )

    return {
        products,
        users,
        colorsIndex,
        categoriesIndex,
        productAvgAgeNorm,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        // price + age + colors + categories
        dimentions: 2 + categories.length + colors.length
    }
}

const oneHotWeighted = (index, length, weight) => {
    return tf.oneHot(index, length).cast('float32').mul(weight)
}

function encodeProduct(product, context) {
    // Normalizando dados para ficar de 0 a 1
    // Aplicar o peso na recomendação
    const price = tf.tensor1d([
        normalize(product.price, context.minPrice, context.maxPrice) * WEIGHTS.price
    ])

    const age = tf.tensor1d([
        (
            context.productAvgAgeNorm[product.name] ?? 0.5
        ) * WEIGHTS.age
    ])

    const category = oneHotWeighted(context.categoriesIndex[product.category],
        context.numCategories, WEIGHTS.category
    )

    const color = oneHotWeighted(context.colorsIndex[product.color],
        context.numColors, WEIGHTS.color
    )

    return tf.concat1d([price, age, category, color])
}

function encodeUser(user, context) {
    if (user.purchases.length) {
        return tf.stack(
            user.purchases.map(product => encodeProduct(product, context))
        ).
            mean(0)
            .reshape([
                1,
                context.dimentions
            ])
    }

    return tf.concat1d([
        tf.zeros([1]), // preço é ignorado
        tf.tensor1d([
            normalize(user.age, context.minAge, context.maxAge) * WEIGHTS.age
        ]),
        tf.zeros([context.numCategories]), //categoria
        tf.zeros([context.numColors]), //cores
    ]).reshape([1, context.dimentions])
}

function createTrainingData(context) {
    const inputs = []
    const labels = []
    context.users
        .filter(u => u.purchases.length)
        .forEach(user => {
            const userVector = encodeUser(user, context).dataSync()
            context.products.forEach(p => {
                const productVector = encodeProduct(p, context)
                    .dataSync()
                const label = user.purchases.some(
                    purchase => purchase.name === p.name ?
                        1 : 0
                )
                //compibar user + product
                inputs.push([...userVector, ...productVector])
                labels.push(label)
            })
        })
    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimesion: context.dimentions * 2
        //tamanho é igual a userVector + productVector
    }
}

/**
 * Configura e treina uma rede neural sequencial para classificação binária.
 * 
 * @async
 * @param {Object} trainData - Dados de treinamento da rede neural
 * @param {number} trainData.inputDimesion - Dimensão do vetor de entrada
 * @param {tf.Tensor} trainData.xs - Tensor com os dados de entrada (features)
 * @param {tf.Tensor} trainData.ys - Tensor com os rótulos de saída (labels)
 * 
 * @returns {Promise<tf.Sequential>} Modelo neural treinado pronto para uso
 * 
 * @description
 * Cria uma rede neural com 4 camadas densas (128, 64, 32 unidades e 1 unidade de saída).
 * Utiliza ativação ReLU nas primeiras 3 camadas e sigmoid na camada de saída.
 * O modelo é compilado com otimizador Adam e função de perda binary crossentropy.
 * Durante o treinamento, envia mensagens de progresso via postMessage com a perda e acurácia
 * de cada epoch. O treinamento executa por 100 epochs com batch size de 32.
 */
async function configureNeuralNetAndTrain(trainData) {
    const model = tf.sequential()
    model.add(
        tf.layers.dense({
            inputShape: [trainData.inputDimesion],
            units: 128,
            activation: 'relu'
        })
    )
    model.add(
        tf.layers.dense({
            units: 64,
            activation: 'relu'
        })
    )
    model.add(
        tf.layers.dense({
            units: 32,
            activation: 'relu'
        })
    )
    model.add(
        tf.layers.dense({ units: 1, activation: 'sigmoid' })
    )

    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    })

    await model.fit(trainData.xs, trainData.ys, {
        epochs: 100,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch,
                    loss: logs.loss,
                    accuracy: logs.acc // || logs.accuracy || 0
                });
            }
        }
    });

    return model
}

// Função responsável por treinar o modelo de recomendação.
// 1. Recebe a lista de usuários como parâmetro.
// 2. Busca os produtos do arquivo JSON.
// 3. Cria o contexto com índices e normalizações necessárias.
// 4. Codifica os produtos em vetores numéricos.
// 5. Gera os dados de treino (inputs e labels) combinando usuários e produtos.
// 6. Treina a rede neural com esses dados.
// 7. Atualiza o progresso e notifica quando o treinamento termina.
async function trainModel({ users }) {
    console.log('Training model with users:', users);
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 1 } });
    const products = await (await fetch('/data/products.json')).json()

    const context = makeContext(products, users)

    context.productVectors = products.map(product => {
        return {
            name: product.name,
            meta: { ...product },
            productVector: encodeProduct(product, context).dataSync()
        }
    })

    _globalCtx = context;
    const trainData = createTrainingData(context);

    // Aguarda o modelo finalizar o treinamento antes de atualizar o progresso
    _model = await configureNeuralNetAndTrain(trainData);

    // Notificação final do progresso após termino do treinamento
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}

function recommend({ user }) {
    if (!_model) return;
    const context = _globalCtx
    // Converta o usuario fornecido no vetor de features codificadas(preço ignorado, idade normalizada, categorias e cores ignoradas)
    // Isso transforma as informações do usuario no mesmo formato numerico que foi usado para treinar o modelo
    const userVector = encodeUser(user, context).dataSync()

    // Em aplicações reais:
    // Armazene todos os vetores de produtos em um banco de dados vetorial(postgress, Neo4j ou Pinecone)
    // Consulta: Encontre os 200 produtos mais proximos do vetor do usuario
    // Execute _model.predict() apenas nesses produtos

    // Crie pares de entrada: para cada produto, concate o vetor do usuario com o vetor codificado do produto
    // Por que? O modelo prevê o "score de compatibilidade" para cada par(usuario, produto)
    const inputs = context.productVectors.map(({ productVector }) => {
        return [
            ...userVector,
            ...productVector
        ]
    })

    // Converta todos esses pares(usuario,produto) em um uinico tensor
    // Formato: [numProdutos,inputDim]
    const inputTensor = tf.tensor2d(inputs)

    // Rode a rede neural treinada em todos os pares(usuario,produto) de uma vez
    // O resultado é uma pontuação para cada produto entre 0 e 1
    // Quanto maior, maior a probabilidade do usuario querer aquele produto.
    const predictions = _model.predict(inputTensor)

    // Extrai as pontuações para um array js normal
    const scores = predictions.dataSync()
    const recommendations = context.productVectors.map((item, index) => {
        return {
            ...item.meta,
            name: item.name,
            score: scores[index] // previsão do modelo para este produto
        }
    })

    const sortedItems = recommendations
        .sort((a, b) => b.score - a.score)


    // Envia a lista ordenada de produtos recomendados para a thread principal (a UI pode exibi-los agora)
    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: sortedItems
    });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: recommend,
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};