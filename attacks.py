import torch
import torch.optim as optim

def cw_attack(model, image, label, target_label, c=1e-4, kappa=0, lr=0.01, num_iter=100):
    """
    Implementazione dell'attacco Carlini & Wagner (C&W).
    
    :param model: Il modello da attaccare (torch.nn.Module)
    :param image: L'immagine originale (tensor)
    :param label: La classe vera dell'immagine (tensor)
    :param target_label: La classe target desiderata (tensor)
    :param c: Peso del termine di penalità per la perdita (float)
    :param kappa: Confidenza minima per l'inganno (float, opzionale)
    :param lr: Learning rate per l'ottimizzatore (float)
    :param num_iter: Numero di iterazioni per l'ottimizzazione (int)
    :return: Immagine avversariale
    """

    # Mettiamo il modello in modalità valutazione
    model.eval()

    # Creiamo una variabile ottimizzabile (w) per l'immagine perturbata
    w = torch.zeros_like(image, requires_grad=True)

    # Definiamo l'ottimizzatore
    optimizer = optim.Adam([w], lr=lr)

    # Funzione di perdita per il termine L2
    def l2_loss(x, x_adv):
        return torch.sum((x - x_adv) ** 2)

    # Softmax cross-entropy modificata con kappa
    def f(output, target_label, kappa):
        target_logits = output[0, target_label]
        other_logits = torch.max(torch.cat((output[0, :target_label], output[0, target_label + 1:])))
        return torch.clamp(other_logits - target_logits + kappa, min=0)

    for _ in range(num_iter):
        # Calcolo dell'immagine perturbata
        perturbed_image = 0.5 * (torch.tanh(w) + 1)  # Mapping w -> intervallo [0, 1]

        # Forward pass con l'immagine perturbata
        output = model(perturbed_image)

        # Calcolo della perdita
        loss1 = l2_loss(image, perturbed_image)  # Perdita L2
        loss2 = c * f(output, target_label, kappa)  # Perdita per inganno (confidenza)
        loss = loss1 + loss2

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Ritorna l'immagine avversariale finale
    perturbed_image = 0.5 * (torch.tanh(w) + 1)
    return perturbed_image.detach()

def pgd_attack(image, label, model, epsilon, alpha, num_iter, criterion):
    """
    :param image: The input image (tensor)
    :param label: The true label for the image (tensor) O CHE ATT.
    :param model: The model being attacked (torch.nn.Module)
    :param epsilon: The maximum perturbation
    :param alpha: The step size for each iteration
    :param num_iter: The number of iterations for the attack
    :param criterion: The loss function (e.g., nn.CrossEntropyLoss)
    :return: The adversarial image after applying PGD
    Alpha: Quanto velocemente aggiorniamo l'immagine verso una perturbazione avversariale.
    Epsilon: Quanto lontano possiamo spingerci rispetto all'immagine originale.
    Iterations: Quante volte aggiorniamo l'immagine per massimizzare l'effetto dell'attacco.
    """

    # Set the model to evaluation mode
    model.eval()

    # Ensure gradient computation for the original image
    image.requires_grad = True

    for _ in range(num_iter):
        # Forward pass
        output = model(image)

        # Compute the loss
        loss = criterion(output, label)

        # Zero previous gradients
        model.zero_grad()

        # Backward pass: calculate gradients
        loss.backward()

        # Get the gradient of the loss with respect to the image
        grad = image.grad.data

        # Update the perturbed image using the gradient
        perturbed_image = image + alpha * grad.sign()

        # Clip the perturbation to ensure the image remains within valid bounds
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        # Project the perturbed image to the epsilon ball around the original image
        perturbed_image = torch.max(torch.min(perturbed_image, image + epsilon), image - epsilon)

        # Detach and re-enable gradient tracking for the next iteration
        image = perturbed_image.clone().detach().requires_grad_(True)

    return perturbed_image




def fgsm_attack(image, label, model, epsilon, criterion):

    # Imposta il modello in modalità valutazione
    model.eval()
    
    # Abilita il calcolo del gradiente
    image.requires_grad = True

    # Esegui il forward pass
    output = model(image)  
    #get cam from model
    
    # Calcola la perdita
    loss = criterion(output, label)
    
    # Azzerare i gradienti precedenti
    model.zero_grad()
    
    # Calcola il gradiente della perdita rispetto all'immagine
    loss.backward()
    
    # Prendi il segno del gradiente
    grad = image.grad.data
    
    # Crea la nuova immagine modificata
    perturbed_image = image + epsilon * grad.sign()
    
    # Assicurati che i valori dell'immagine siano compresi tra 0 e 1
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image